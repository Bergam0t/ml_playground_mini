import base64
import hashlib
import json
from pathlib import Path

import streamlit as st
import pandas as pd

import calls
import charts
import costs
import explain
import weekly
from demo_data import DATA_DIR, load_forecast  # DEMO: synthetic data replaces SQL
from mobile_nav import add_mobile_nav

HSMA_URL = "https://hsma.co.uk"

st.set_page_config(
    page_title="Outpatient DNA Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.logo(
    "resources/Chalky HSMA-07-wide white logo tall with tagline - new tagline.png",
    size="large",
    link=HSMA_URL,
)
st.markdown(f"<style>{Path('style.css').read_text()}</style>", unsafe_allow_html=True)

# HSMA display font, embedded as base64 so it loads without static file serving.
font_b64 = base64.b64encode(
    Path("resources/fonts/hsma-display/HSMADisplay-Regular.woff2").read_bytes()
).decode()
st.markdown(
    f"""<style>
    @font-face {{
        font-family: "HSMA Display";
        src: url(data:font/woff2;base64,{font_b64}) format("woff2");
    }}
    </style>""",
    unsafe_allow_html=True,
)


# DEMO: data comes from demo_data.py instead of vw_Combined_Future_OP_Forecast
@st.cache_data
def load_data():
    return load_forecast()


# SHAP values and model details, precomputed by build/train_and_explain.py
@st.cache_data
def load_shap():
    return pd.read_csv(DATA_DIR / "shap_values.csv").set_index("row_id")


# Past appointments the model didn't train on, with what really happened (for tab 3)
@st.cache_data
def load_past():
    return pd.read_csv(DATA_DIR / "past_predictions.csv")


@st.cache_data
def load_model_meta():
    return json.loads((DATA_DIR / "model_meta.json").read_text(encoding="utf-8"))


# Fewer filtered appointments than this and their factor chart is too noisy to show
MIN_GROUP_FOR_CHART = 30

# The "How to read this chart" steps; the numbers match the badges on charts.guide_waterfall()
GUIDE_STEPS = [
    (
        "<b>Start at the top: the average appointment.</b> This is the model's predicted risk "
        "of not attending, averaged over all appointments."
    ),
    (
        "<b>Each bar is one fact about this appointment.</b> Red pushes the risk of not "
        "attending up, blue pushes it down. The number is how far it moves, in percentage "
        "points. The biggest come first."
    ),
    (
        "<b>The bottom bar is this appointment's predicted risk.</b> The bars add up from the "
        "average to it. They show what the model relied on, not what causes a missed appointment."
    ),
]


# Risk bands, the yes/no forecast and the high-risk flag, set for a DNA rate of about 8%
# (demo_data.DNA_RATE). The data's DNA_Forecast is a risk of FORECAST_MIN or more (set in
# build/train_and_explain.py, which prints how the forecasts fall into all of these).
FORECAST_MIN = 0.20  # about two and a half times the average; the live model's 50%
HIGH_RISK_MIN = 0.40  # the Very High band


def DNA_Risk(prob):
    if prob < 0.05:
        return "Low"
    elif prob < FORECAST_MIN:
        return "Medium"
    elif prob < HIGH_RISK_MIN:
        return "High"
    else:
        return "Very High"


def Confidence_Score_Txt(confidence):
    if confidence < 0.5:
        return "Low"
    elif confidence < 0.8:
        return "Medium"
    elif confidence < 0.9:
        return "High"
    else:
        return "Very High"


# Cell colours for the model's forecast columns on the appointments table. Dark shades
# with white text so they read on the dark theme; the patient columns stay uncoloured.
EXPECTED_DNA_COLOURS = {"Yes": "#8b1e2d", "No": "#1e5631"}
# Confidence is neither good nor bad, so it uses a single-hue teal ramp, not red/green
CONFIDENCE_COLOURS = {
    "Low": "#1c3436", "Medium": "#164f53", "High": "#0f6a70", "Very High": "#07858d",
}
RISK_COLOURS = {
    "Low": "#1e5631", "Medium": "#7a6a12", "High": "#9a4a12", "Very High": "#8b1e2d",
}


def forecast_cell_styles(df):
    """CSS for each cell of the appointments table: only the forecast columns are coloured."""

    def css(colour):
        return f"background-color: {colour}; color: #ffffff;"

    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles["Expected DNA"] = df["Expected DNA"].map(EXPECTED_DNA_COLOURS).map(css)
    styles["Confidence"] = df["Confidence"].map(CONFIDENCE_COLOURS).map(css)
    # Same bands as the sidebar's "DNA Risk" filter
    styles["Probabilty"] = (df["Probabilty"] / 100).map(DNA_Risk).map(RISK_COLOURS).map(css)
    return styles


# Appointment tables (the Appointments tab, and a day's list on the Weekly summary tab):
# source column -> display name, in display order
APPOINTMENT_FIELDS = {
    "Activity_Date": "Appointment Date",
    "MRN": "Patient ID",
    "Mobile Telephone": "Mobile Number",
    "Specialty": "Specialty",
    "Appointment Location Text": "Location",
    "Appointment Type": "Appointment Type",
    "DNA_Forecast_Text": "Expected DNA",
    "Confidence": "Confidence",
    "Probabilty": "Probabilty",
}

# Fixed column widths (about 1,110px in total) so the whole table fits a 1,400px
# main area; the grid stretches them to fill any space left over.
APPOINTMENT_COLUMNS = {
    "Appointment Date": st.column_config.DateColumn(
        "Appointment Date", format="DD/MM/YYYY", width=110
    ),
    "Patient ID": st.column_config.TextColumn(width=95),
    "Mobile Number": st.column_config.TextColumn(width=120),
    #'Mobile Number (GP Record)': st.column_config.TextColumn(),
    "Specialty": st.column_config.TextColumn(width=165),
    "Location": st.column_config.TextColumn(width=175),
    "Appointment Type": st.column_config.TextColumn(width=130),
    "Expected DNA": st.column_config.TextColumn(
        width=115,
        help=f"Model forecast: will the patient miss this appointment? Yes for a predicted risk "
        f"of {FORECAST_MIN:.0%} or more. Red Yes, green No.",
    ),
    "Confidence": st.column_config.TextColumn(
        width=105,
        help="Model forecast: how clear-cut the yes-or-no call is. Low for risks near 50%. "
        "Darker teal is less clear-cut.",
    ),
    "Probabilty": st.column_config.NumberColumn(
        "DNA Risk",
        format="%.2f%%",
        width=95,
        help="Model forecast: probability of DNA. Colour shows the risk band, "
        "green Low to red Very High.",
    ),
}


def appointment_table(df, by_patient=False):
    """Appointments to show in a table, with the forecast columns coloured.

    `by_patient` puts the patient columns first and shades every other patient's rows,
    for a table sorted so each patient's appointments are together.
    """
    fields = list(APPOINTMENT_FIELDS)
    if by_patient:
        patient = ["MRN", "Mobile Telephone"]
        fields = patient + [f for f in fields if f not in patient]
    shown = df[fields].rename(columns=APPOINTMENT_FIELDS)
    styler = shown.style.apply(forecast_cell_styles, axis=None)
    return styler.apply(patient_bands, axis=None) if by_patient else styler


def patient_bands(df):
    """Shade every other patient's rows (outside the coloured forecast columns)."""
    patient_no = (df["Patient ID"] != df["Patient ID"].shift()).cumsum()
    plain = [c for c in df.columns if c not in ("Expected DNA", "Confidence", "Probabilty")]
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles.loc[patient_no % 2 == 0, plain] = "background-color: #2e2c2c;"
    return styles


def demo_notice():
    st.warning(
        "DEMONSTRATION VERSION - all data shown is synthetic. "
        "No real patients or Trust data are displayed."
    )


# Streamlit app
# The tabs come first so style.css can pin them to the top as a navbar; the app title
# is drawn at the left of that bar by style.css.
# Dropdown in place of the tab pills on narrower screens (shown by style.css)
add_mobile_nav()
tab_appts, tab_weekly, tab_calls = st.tabs(["Appointments", "Weekly summary", "Who would we phone?"])
# st.subheader("AI Forecast of the Risk of a Patient Not Attending an Appointment")

# Load data
# st.write("Loading data...")
data = load_data()
# st.write("Data loaded successfully!")

# Create dummy data for presentation
# data['MRN'] = ['MRN' + str(i).zfill(6) for i in range(1, len(data) + 1)]
# data['Mobile Telephone'] = ['07' + str(100000000 + i)[1:] for i in range(1, len(data) + 1)]

data["Risk"] = data["DNA_Prob_Forecast"].apply(DNA_Risk)
data["Confidence"] = data["Confidence_Score"].apply(Confidence_Score_Txt)

# Flagged high-risk: the Very High band, stricter than the yes/no forecast as in the live app.
# (The live app's flag, risk over 70% and confidence over 70%, would catch almost nobody at
# an 8% DNA rate.)
data["High_Risk_DNA"] = (data["DNA_Prob_Forecast"] >= HIGH_RISK_MIN).astype(int)
data["Probabilty"] = data["DNA_Prob_Forecast"] * 100


# NIHR logo, directly under the HSMA logo in the sidebar header (ported from the
# geographic playground mini). Keyed so style.css can size it and hide the image's
# hover "fullscreen" button.
with st.sidebar.container(key="nihr_logo"):
    st.image(
        "resources/NIHR_APPLIED_RESEARCH_COLLABORATION_Logo_Horizontal_RGB_WHITE.png",
        width="stretch",
        link="https://arc-sw.nihr.ac.uk/",
    )

# Filters
st.sidebar.header("Filters")
st.sidebar.caption("Choose which upcoming appointments the app shows.")
# Start with the full data
filtered_data = data.copy()

# Facts about the appointment first, then the model's forecast for it
st.sidebar.subheader("Appointment details")

# Allow the user to filter by Site
reporting_site_options = sorted(
    filtered_data["Reporting_Site"].dropna().unique()
)
selected_reporting_site = st.sidebar.selectbox(
    "Select Reporting Site", options=["All"] + list(reporting_site_options)
)

if selected_reporting_site != "All":
    filtered_data = filtered_data[
        filtered_data["Reporting_Site"] == selected_reporting_site
    ]


# Specialty filter options (always from full data)
specialty_options = sorted(filtered_data["Specialty"].dropna().unique())
selected_specialty = st.sidebar.selectbox(
    "Select Specialty", options=["All"] + list(specialty_options)
)

if selected_specialty != "All":
    filtered_data = filtered_data[
        filtered_data["Specialty"] == selected_specialty
    ]

# Appointment Type options (from filtered data)
appointment_type_options = sorted(
    filtered_data["Appointment Type"].dropna().unique()
)
selected_appointment_type = st.sidebar.selectbox(
    "Select Appointment Type", options=["All"] + list(appointment_type_options)
)

if selected_appointment_type != "All":
    filtered_data = filtered_data[
        filtered_data["Appointment Type"] == selected_appointment_type
    ]

# Appointment Location options (from filtered data)
appointment_location_options = sorted(
    filtered_data["Appointment Location Text"].dropna().unique()
)
selected_appointment_location = st.sidebar.selectbox(
    "Select Appointment Location",
    options=["All"] + list(appointment_location_options),
)

if selected_appointment_location != "All":
    filtered_data = filtered_data[
        filtered_data["Appointment Location Text"]
        == selected_appointment_location
    ]

if filtered_data.empty:  # DEMO: avoid crashing on empty filter combinations
    st.info("No appointments match the selected filters.")
    st.stop()

# Date filter (from filtered data)
min_date = pd.to_datetime(filtered_data["Activity_Date"]).min()
max_date = pd.to_datetime(filtered_data["Activity_Date"]).max()
selected_date = st.sidebar.date_input(
    "Select Date",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_date, (tuple, list)) and len(selected_date) == 2:
    start_date, end_date = selected_date
    filtered_data = filtered_data[
        (
            pd.to_datetime(filtered_data["Activity_Date"])
            >= pd.to_datetime(start_date)
        )
        & (
            pd.to_datetime(filtered_data["Activity_Date"])
            <= pd.to_datetime(end_date)
        )
    ]

st.sidebar.divider()
st.sidebar.subheader("Model forecast")

# DNA Risk options (from filtered data)
risk_options = ["Low", "Medium", "High", "Very High"]
available_risks = [
    r for r in risk_options if r in filtered_data["Risk"].unique()
]
selected_dna_risk = st.sidebar.selectbox(
    "Select DNA Risk",
    options=["All"] + available_risks,
    # Bands from DNA_Risk()
    help="The model's predicted risk that the patient won't attend.  \n"
    "**Low:** under 5%  \n**Medium:** 5% to 20%  \n**High:** 20% to 40%  \n"
    "**Very High:** 40% or more  \n"
    "About 8% of appointments are missed. High and Very High (20% or more, about two and a half "
    "times the average) are forecast as DNAs; Very High is also flagged high-risk.",
)

if selected_dna_risk != "All":
    filtered_data = filtered_data[filtered_data["Risk"] == selected_dna_risk]


# --- NEW: Filter for Confidence Level ---
confidence_options = ["Low", "Medium", "High", "Very High"]
available_confidences = [
    c for c in confidence_options if c in filtered_data["Confidence"].unique()
]
selected_confidence = st.sidebar.selectbox(
    "Select Confidence Level",
    options=["All"] + available_confidences,
    # Bands from Confidence_Score_Txt()
    help="How clear-cut the model's yes-or-no call is: the gap between its chance that the "
    "patient won't attend and its chance that they will. It follows from the risk.  \n"
    "**Low:** under 50% (risk 25% to 75%)  \n"
    "**Medium:** 50% to 80% (risk 10% to 25% or 75% to 90%)  \n"
    "**High:** 80% to 90% (risk 5% to 10% or 90% to 95%)  \n"
    "**Very High:** 90% or more (risk under 5% or over 95%)",
)

if selected_confidence != "All":
    filtered_data = filtered_data[
        filtered_data["Confidence"] == selected_confidence
    ]
# --- END of new filter ---

# Cost assumptions: not filters, so they stay out of filter_key below and editing them
# never clears a table selection
st.sidebar.divider()
st.sidebar.subheader("Cost assumptions")
band5 = costs.admin_hourly_cost(**costs.BAND5)
with st.sidebar.expander("Calls and missed appointments", icon=":material/payments:"):
    hourly = st.number_input(
        "Admin staff cost per hour (£)",
        min_value=0.0,
        value=round(band5["hourly"], 2),
        step=0.5,
        format="%.2f",
        key="cost_hourly",
        help=f"Default: a band 5 admin. Salary £{band5['salary']:,.0f} (Agenda for Change "
        f"2025/26, middle pay point) plus employer National Insurance £{band5['ni']:,.0f} and "
        f"pension £{band5['pension']:,.0f}: £{band5['total']:,.0f} a year, over about "
        f"{band5['hours']:,.0f} working hours once leave, bank holidays, sickness and training "
        "are taken off.",
    )
    minutes_per_call = st.number_input(
        "Minutes per call",
        min_value=0.5,
        value=float(costs.MINUTES_PER_CALL),
        step=0.5,
        format="%.1f",
        key="cost_minutes",
        help="On average, including unanswered attempts and noting the outcome.",
    )
    cost_per_call = hourly * minutes_per_call / 60
    st.caption(f"So each call costs about **£{cost_per_call:.2f}** in staff time.")

    share_lost = st.slider(
        "Share of a missed appointment's cost really lost",
        0,
        100,
        100,
        step=5,
        format="%d%%",
        key="cost_share_lost",
        help="Lower this if some empty slots get re-used, for example by overbooking or a "
        "clinician catching up on other work.",
    ) / 100
    dna_costs = st.data_editor(
        costs.dna_cost_table(),
        column_config={
            "Specialty": st.column_config.TextColumn(disabled=True),
            "New": st.column_config.NumberColumn("New (£)", min_value=0, format="£%d"),
            "Follow Up": st.column_config.NumberColumn("Follow up (£)", min_value=0, format="£%d"),
        },
        hide_index=True,
        key="cost_dna_table",
        alt="Cost of a missed appointment by specialty and appointment type",
    )
    st.caption(
        "Cost of a missed appointment. Illustrative values, loosely in line with NHS Payment "
        "Scheme outpatient prices: replace them with your Trust's own costs."
    )

if filtered_data.empty:
    st.info("No appointments match the selected filters.")
    st.stop()

filtered_data["DNA_Cost"] = costs.appointment_costs(filtered_data, dna_costs, share_lost)

# Remap DNA_Forecast values from 1/0 to Yes/No
filtered_data["DNA_Forecast_Text"] = filtered_data["DNA_Forecast"].replace(
    {1: "Yes", 0: "No"}
)

# A new key whenever the filters change, so an old table selection never points at
# a different appointment (or day) in the newly filtered table.
filter_key = hashlib.md5(
    repr(
        (
            selected_reporting_site,
            selected_specialty,
            selected_appointment_type,
            selected_appointment_location,
            selected_dna_risk,
            selected_confidence,
            selected_date,
        )
    ).encode()
).hexdigest()[:10]

# ---------------------------------------------------------------- tab 1
with tab_appts:
    demo_notice()

    st.markdown(
        "**This app helps outpatient teams see which upcoming appointments patients are most "
        "likely to miss** (a DNA, or 'did not attend'), so they can plan where reminder calls or "
        "other support might help most. A machine-learning model, trained on past appointments, "
        "gives each one a predicted risk of not attending and a confidence score. The forecasts "
        "are estimates to support staff judgement, not decisions.\n\n"
        "About 8% of appointments are missed, so most predicted risks are low. **The model "
        "forecasts a DNA for a risk of 20% or more** (about two and a half times the average), and "
        "**flags a risk of 40% or more as high risk**.\n\n"
        "**Confidence** is how clear-cut the model's yes-or-no call is: the gap between its chance "
        "that the patient won't attend and its chance that they will. A 90% risk (90% against 10%) "
        "gives 80% confidence; a 50% risk gives 0%, where the model could go either way. So "
        "confidence follows from the risk: it is Low for any risk between 25% and 75%, and high "
        "for very low risks as well as very high ones. Most appointments here have high confidence, "
        "because the model is fairly sure the patient will attend.\n\n"
        "This tab lists the upcoming appointments with the model's forecast for each, and explains "
        "what lies behind any one forecast."
    )

    # Summary of the filtered appointments
    with st.container(horizontal=True):
        st.metric(
            "Appointments (filtered)",
            f"{len(filtered_data):,}",
            icon=":material/calendar_month:",
            border=True,
        )
        st.metric(
            "Forecast DNAs",
            f"{int(filtered_data['DNA_Forecast'].sum()):,}",
            icon=":material/event_busy:",
            border=True,
            help=f"Appointments the model forecasts as a DNA: a predicted risk of {FORECAST_MIN:.0%} "
            "or more (the 'Expected DNA' column). Not all of them will be missed, and some "
            "lower-risk appointments will be.",
        )
        st.metric(
            "High-risk flagged",
            f"{filtered_data['High_Risk_DNA'].mean() * 100:.1f}%",
            icon=":material/warning:",
            border=True,
            help=f"Share of appointments with a predicted risk of {HIGH_RISK_MIN:.0%} or more "
            "(the Very High band).",
        )
        st.metric(
            "Expected cost of missed appointments",
            f"£{costs.expected_dna_cost(filtered_data):,.0f}",
            icon=":material/payments:",
            border=True,
            help="Each appointment's predicted risk of not attending times the cost of missing "
            "it, added up. The costs are assumptions you can change under **Cost assumptions** "
            "in the sidebar.",
        )

    st.info(
        "**Click a row in the table** to see what made the model give that "
        "appointment its risk of not attending. Until you do, the chart below shows what the "
        "model relies on most. The coloured columns are the model's forecast.",
        icon=":material/touch_app:",
    )

    # Display filtered data
    table = st.dataframe(
        appointment_table(filtered_data),
        column_config=APPOINTMENT_COLUMNS,
        hide_index=True,
        alt="Upcoming appointments with the forecast risk of not attending",
        on_select="rerun",
        selection_mode="single-row",
        key=f"appointments_{filter_key}",
    )

    # Selected positions refer to the rows as passed in, whatever sorting the user applies
    selected_rows = [r for r in table.selection.rows if r < len(filtered_data)]
    if selected_rows:
        row = filtered_data.iloc[selected_rows[0]]
        meta = load_model_meta()
        expl = explain.patient_explanation(
            row, load_shap().loc[row["row_id"]], meta["base_value_pp"]
        )

        st.subheader(
            f"Why patient {row['MRN']} on {row['Activity_Date']:%d/%m/%Y} has a "
            f"{expl['fx']:.0f}% risk of not attending"
        )
        st.markdown(
            f"**The model predicts a {expl['fx']:.0f}% risk that this patient won't attend**, "
            f"compared with about **{expl['base']:.0f}%** for the average appointment."
        )

        # Starts open; collapse it once you have the idea (it stays collapsed between patients)
        with st.expander(
            "How to read this chart", expanded=True, key="chart_guide", icon=":material/help:"
        ):
            # Keyed so style.css stacks the example above the steps on narrower screens
            example_col, steps_col = st.container(key="reflow_stack_guide").columns(
                [4, 5], vertical_alignment="center"
            )
            example_col.plotly_chart(
                charts.guide_waterfall(),
                key="guide_waterfall",
                width="content",
                config={"staticPlot": True},
            )
            example_col.caption("A made-up example.")
            steps_col.markdown(
                "\n".join(
                    f"<div class='guide-step'><span class='guide-num'>{n}</span><div>{text}</div></div>"
                    for n, text in enumerate(GUIDE_STEPS, start=1)
                ),
                unsafe_allow_html=True,
            )

        st.plotly_chart(
            charts.waterfall(expl),
            key=f"waterfall_{row['row_id']}",
            width="stretch",
            config={"displayModeBar": False},
        )

        top = expl["shown"][:2]
        st.markdown(
            f"The biggest contributions are **{top[0]['label']}** ({top[0]['value']:+.1f} points) and "
            f"**{top[1]['label']}** ({top[1]['value']:+.1f} points)."
        )
        st.info(
            "A predicted risk is an estimate based on similar appointments in past data, not a "
            "forecast of what this one person will do. The bars show what the model relied on, "
            "not what causes missed appointments: changing one of these facts would not "
            f"necessarily change whether the patient attends. The model is reasonably but not "
            f"perfectly accurate (AUC {meta['auc_test']:.2f} on past appointments it hadn't seen)."
        )
    else:
        # No row selected: what the model relies on across all upcoming appointments, and,
        # when the sidebar filters narrow the table enough, for the filtered ones beside it
        meta = load_model_meta()
        shap_all = load_shap()
        n = len(filtered_data)
        is_filtered = n < len(data)
        # Not for the forecast filters: they pick appointments by the very pushes being
        # charted (Very High risk = big pushes up), so the comparison would be circular
        forecast_filtered = selected_dna_risk != "All" or selected_confidence != "All"
        show_filtered = is_filtered and n >= MIN_GROUP_FOR_CHART and not forecast_filtered

        st.subheader("What the model relies on most")
        st.markdown(
            "Each bar shows how much a factor moves an appointment's predicted risk of not "
            "attending, on average, **whether it pushes the risk up or down**. A longer bar means "
            "the model relies on that factor more. Hover over a bar for the figure."
        )
        st.plotly_chart(
            charts.importance_bars(
                explain.group_importance(shap_all.loc[data["row_id"]]),
                f"All {len(data):,} upcoming appointments",
                explain.group_importance(shap_all.loc[filtered_data["row_id"]]) if show_filtered else None,
                f"The {n:,} appointments in your filters",
            ),
            key=f"importance_{filter_key}",
            width="stretch",
            config={"displayModeBar": False},
        )

        if show_filtered:
            notes = [
                (
                    f"On average the model predicts a **{filtered_data['DNA_Prob_Forecast'].mean():.0%}** "
                    "risk of not attending for the appointments in your filters, compared with "
                    f"**{data['DNA_Prob_Forecast'].mean():.0%}** for all upcoming appointments."
                )
            ]
            same_value = [
                label
                for label, choice in [
                    ("site", selected_reporting_site),
                    ("specialty", selected_specialty),
                    ("appointment type", selected_appointment_type),
                ]
                if choice != "All"
            ]
            if same_value:
                listed = " and ".join(same_value)
                notes.append(
                    f"Every appointment in your filters has the same {listed}, so "
                    f"{'that bar shows' if len(same_value) == 1 else 'those bars show'} how much "
                    f"that one {listed} moves the risk."
                )
            st.markdown("\n".join(f"- {note}" for note in notes))
        elif forecast_filtered:
            st.caption(
                "No chart for your filters while a Model forecast filter (DNA Risk or Confidence) "
                "is set: those filters pick appointments by the forecast this chart explains, so "
                "the comparison would go round in circles."
            )
        elif is_filtered:
            st.caption(
                f"Only {n:,} appointment{'s' if n != 1 else ''} match your filters: too few for a "
                f"reliable chart of their own (it needs at least {MIN_GROUP_FOR_CHART})."
            )
        st.info(
            "These bars show what the model relied on, not what causes missed appointments: "
            "changing one of these facts would not necessarily change whether a patient attends. "
            f"The model is reasonably but not perfectly accurate (AUC {meta['auc_test']:.2f} on "
            "past appointments it hadn't seen)."
        )

# ---------------------------------------------------------------- tab 2
with tab_weekly:
    demo_notice()

    st.subheader("Weekly Forecast Summary")
    st.markdown(
        "How many of the upcoming appointments are forecast to be missed, by week and day, so you "
        "can see which weeks and days to focus on."
    )

    measure = st.segmented_control(
        "Count as a missed appointment",
        list(weekly.MEASURES),
        default="High-risk flagged",
        key="weekly_measure",
        help="  \n".join(f"**{name}:** {m.help}" for name, m in weekly.MEASURES.items()),
    ) or "High-risk flagged"
    m = weekly.MEASURES[measure]
    column = m.column
    h = weekly.headline(filtered_data, column)

    # Headline figures for the appointments in the sidebar filters
    with st.container(horizontal=True):
        st.metric(
            "Appointments",
            f"{h['n']:,}",
            icon=":material/calendar_month:",
            border=True,
            help=f"Over {h['n_weeks']} week{'s' if h['n_weeks'] != 1 else ''}, in the sidebar filters.",
        )
        st.metric(
            m.count_label,
            f"{h['total']:,.0f} ({h['overall']:.0%})",
            icon=":material/event_busy:",
            border=True,
            help=m.help,
        )
        # The first week in the filters: usually next week, the one there is still time to act on
        vs_average = round(h["first_week_total"] - h["weekly_average_total"])
        st.metric(
            weekly.week_title(h["first_week"]),
            f"{h['first_week_total']:,.0f} ({h['first_week_rate']:.0%})",
            f"{vs_average:+d} vs weekly average",
            delta_color="inverse",
            icon=":material/event_upcoming:",
            border=True,
            help=f"{m.count_label} in the week ending {weekly.week_label(h['first_week'], year=True)}, "
            f"from {h['first_week_n']:,} appointments. The weekly average is "
            f"{h['weekly_average_total']:.0f}.",
        )
        st.metric(
            "Highest-risk weekday",
            h["top_day"],
            weekly.pts_vs_average(h["top_day_rate"], h["overall"]),
            delta_color="inverse",
            icon=":material/today:",
            border=True,
            help=f"{h['top_day_rate']:.0%} on {h['top_day']}s, against {h['overall']:.0%} overall.",
        )
        expected_cost = costs.expected_dna_cost(filtered_data)
        st.metric(
            "Expected cost a week",
            f"£{expected_cost / h['n_weeks']:,.0f}",
            icon=":material/payments:",
            border=True,
            help=f"£{expected_cost:,.0f} over {h['n_weeks']} week{'s' if h['n_weeks'] != 1 else ''}. "
            "Each appointment's predicted risk of not attending times the cost of missing it, "
            "whichever count is chosen above. The costs are assumptions you can change under "
            "**Cost assumptions** in the sidebar.",
        )
        rp = weekly.repeat_patients(filtered_data, m)
        # A bordered container rather than the metric's own border, so the button sits in the card
        with st.container(border=True, gap="small"):
            st.metric(
                m.repeat_label,
                f"{rp['patients']:,}",
                f"{rp['appointments']:,} appointments between them"
                + ("" if m.flag else f", about {rp['expected']:.0f} expected to be missed"),
                delta_color="off",
                delta_arrow="off",
                icon=":material/group:",
                help="One reminder call could cover several of a patient's appointments. "
                "Counted over all the weeks in the sidebar filters.",
            )
            showing = st.session_state.get("show_repeat_patients", False)
            st.button(
                "Hide patients" if showing else "Show patients",
                icon=":material/visibility_off:" if showing else ":material/visibility:",
                type="tertiary",
                disabled=rp["patients"] == 0,
                on_click=lambda: st.session_state.update(show_repeat_patients=not showing),
            )

    # The repeat patients, one patient's appointments together, soonest appointment first
    if st.session_state.get("show_repeat_patients") and rp["patients"]:
        st.markdown(
            f"**{m.repeat_label}:** {rp['patients']:,} patients, {rp['appointments']:,} "
            "appointments. The patient with the soonest appointment comes first, and "
            "every other patient is shaded."
        )
        st.dataframe(
            appointment_table(rp["rows"], by_patient=True),
            column_config=APPOINTMENT_COLUMNS,
            hide_index=True,
            key=f"repeat_patients_{filter_key}",
        )

    st.markdown(weekly.summary_sentence(h, measure))

    # Week-by-weekday table: newest week first, weekdays with no appointments left out
    counts, sums, rates = weekly.weekly_tables(filtered_data, column)
    low, high = m.thresholds
    week_index = pd.Index(
        [weekly.week_label(w, year=True) for w in counts.index], name="Week ending"
    )
    cell_text = pd.DataFrame(
        {
            d: [
                f"{c} · {s:.0f} ({r:.0%})" if c else "–"
                for c, s, r in zip(counts[d], sums[d], rates[d])
            ]
            for d in counts.columns
        },
        index=week_index,
    )
    cell_styles = rates.map(lambda r: weekly.cell_style(r, low, high)).set_axis(week_index)

    st.caption(
        f"Each cell: appointments · {m.cell_label} "
        f"({'rate' if m.flag else 'average predicted risk'}). "
        f"Green below {low:.0%}, red above {high:.0%}."
    )
    st.info(
        "**Click a day in the table** to see its appointments, highest risk first.",
        icon=":material/touch_app:",
    )
    week_table = st.dataframe(
        cell_text.style.apply(lambda _: cell_styles, axis=None),
        width="stretch",
        on_select="rerun",
        selection_mode="single-cell",
        key=f"weekly_{filter_key}",
    )

    # Drill down into the clicked day (clicks on the week labels are ignored)
    picked = [
        (r, day) for r, day in week_table.selection.cells
        if r < len(counts) and day in counts.columns
    ]
    if picked:
        r, day = picked[0]
        date = counts.index[r] - pd.Timedelta(days=6 - weekly.DAYS.index(day))
        day_appts = filtered_data[pd.to_datetime(filtered_data["Activity_Date"]) == date]
        day_label = f"{date:%A} {date.day} {date:%B}"

        if m.flag:
            shown = day_appts[day_appts[column] == 1]
            st.subheader(
                f"{day_label}: {len(shown)} of {len(day_appts)} appointments {m.day_phrase}"
            )
        else:
            shown = day_appts
            st.subheader(
                f"{day_label}: {len(day_appts)} appointments, about "
                f"{day_appts[column].sum():.0f} expected to be missed"
            )

        if day_appts.empty:
            st.info("There are no appointments in the sidebar filters on this day.")
        elif shown.empty:
            st.info(f"None of this day's appointments are {m.day_phrase}.")
        else:
            st.dataframe(
                appointment_table(shown.sort_values("DNA_Prob_Forecast", ascending=False)),
                column_config=APPOINTMENT_COLUMNS,
                hide_index=True,
                key=f"weekly_day_{filter_key}",
            )
            st.caption(
                "To plan reminder calls across the weeks, see the **Who would we phone?** tab. "
                "To see why the model gave an appointment its risk, click it on the "
                "**Appointments** tab."
            )

# ---------------------------------------------------------------- tab 3
# Ported from the "Who would we phone?" page of lambda_ml_streamlit_exercise. The counts
# come from past appointments with known outcomes; the upcoming appointments (which follow
# the sidebar filters) only turn the chosen setting into calls per week.
with tab_calls:
    demo_notice()
    meta = load_model_meta()
    past = load_past()
    p_past = past["DNA_Prob_Forecast"].to_numpy()

    st.subheader("Who would we phone?")
    st.markdown(
        "This tab helps weigh up how many reminder calls to make, and who should get them. "
        "The model gives each appointment a **risk** of not attending, not a yes or no. If the team "
        "phones patients beforehand, someone has to choose who gets a call. You can choose by **how many "
        "calls the team can make**, or by **how high the predicted risk must be**."
    )
    st.info(
        "**This is a thought experiment.** We don't know yet who will miss the upcoming appointments, so "
        f"the chart uses {meta['n_past_test']:,} **past** appointments the model didn't learn from. Nobody "
        "in them was phoned. Imagine the model had made its list for them: we can check who it would have "
        "picked against who really did not attend."
    )

    # Upcoming appointments in the sidebar filters, by week
    upcoming_week = pd.to_datetime(filtered_data["Activity_Date"]).dt.to_period("W-SUN")
    per_week = upcoming_week.value_counts()
    n_weeks = len(per_week)

    mode = st.segmented_control(
        "Choose who to phone by",
        ["Calls the team can make", "Predicted risk"],
        default="Calls the team can make",
        key="call_mode",
    ) or "Calls the team can make"

    # The setting for who gets a call, and the guess at what a call changes, side by side.
    # The reflow_docked_ keys here and below let style.css stack rows where the sidebar
    # leaves the page narrow.
    ctrl_left, ctrl_right = st.container(key="reflow_docked_controls").columns(2, gap="large")

    if mode == "Calls the team can make":
        # Slider runs up to the busiest filtered week, rounded up to a multiple of 10. Clamp
        # the kept value first, so changing the filters never leaves it out of range.
        busiest = max(10, int(-(-per_week.max() // 10) * 10))
        if st.session_state.get("calls_per_week", 0) > busiest:
            st.session_state["calls_per_week"] = busiest
        st.session_state.setdefault("calls_per_week", min(100, busiest))
        cap = ctrl_left.slider(
            "Calls the team can make per week",
            0,
            busiest,
            step=10,
            key="calls_per_week",
            help="Each week, the team phones this many of that week's highest-risk appointments "
            "(from the appointments in the sidebar filters).",
        )
        # Each week, phone that week's highest-risk appointments, up to capacity. The past
        # appointments then get the same share of calls, highest risk first.
        rank_in_week = filtered_data["DNA_Prob_Forecast"].groupby(upcoming_week).rank(
            method="first", ascending=False
        )
        upcoming_called = rank_in_week <= cap
        called = calls.called_by_share(p_past, upcoming_called.mean())
        how = (
            "**This is the cut-off.** It isn't chosen directly: it follows from how many calls the team can "
            "make. More calls means a lower bar, so more patients get a call, and more of them are people "
            "who would have come anyway. **Move the slider and watch it change.**"
        )
    else:
        # Risk only: confidence is |2 x risk - 1|, so it adds nothing a risk bar doesn't, and
        # on its own it would also pick the patients the model is confident will attend.
        # Few risks pass 60% at an 8% DNA rate, so the slider stops there, in 1% steps
        risk_min = ctrl_left.slider(
            "Phone if the predicted risk is at least",
            0,
            60,
            round(HIGH_RISK_MIN * 100),
            step=1,
            format="%d%%",
            key="risk_min",
        ) / 100
        called = calls.called_by_threshold(p_past, risk_min)
        upcoming_called = pd.Series(
            calls.called_by_threshold(filtered_data["DNA_Prob_Forecast"], risk_min),
            index=filtered_data.index,
        )
        how = (
            "**Here you set the bar, and the number of calls follows from it.** A lower bar means more calls, "
            f"and more of them are to people who would have come anyway. {HIGH_RISK_MIN:.0%} matches the "
            f"app's high-risk flag; {FORECAST_MIN:.0%} phones everyone forecast as a DNA."
        )

    effect = ctrl_right.slider(
        "Suppose a call gets this share of the people it reaches, who would have missed, to attend instead",
        0,
        100,
        50,
        step=5,
        format="%d%%",
        key="call_effect",
        help="An assumption, not a measurement. Try a few values: 0% means the calls make no difference.",
    )
    ctrl_right.caption(
        "A guess: nobody knows yet how many a call would change. See **What difference might the calls "
        "make?** below."
    )

    out = calls.call_outcomes(called, past["Did_Not_Attend"], p_past)

    # Weekly figures for the upcoming appointments in the sidebar filters. Precision is None
    # when the setting phones nobody, and dna_cost is None when nobody upcoming is phoned.
    weekly_calls = upcoming_called.sum() / n_weeks
    dna_cost = costs.cost_per_dna_reached(filtered_data[upcoming_called])
    has_costs = out["calls"] > 0 and weekly_calls > 0 and dna_cost is not None
    if out["calls"] > 0:
        weekly_reached = weekly_calls * out["precision"]
        weekly_avoided = weekly_reached * effect / 100
    if has_costs:
        econ = costs.call_economics(
            weekly_calls, weekly_reached, effect / 100, cost_per_call, minutes_per_call, dna_cost
        )
        # None when the calls reach nobody who would miss: no effect could pay for them
        be = econ["break_even_effect"]
        be = float("inf") if be is None else be

    with st.container(border=True):
        k1, k2 = st.container(key="reflow_docked_cutoff").columns([0.35, 0.65], gap="large")
        k1.metric("Calls per 1,000 appointments", f"{out['calls']}")
        k2.metric(
            "So we phone anyone with a predicted risk of at least"
            if mode == "Calls the team can make"
            else "Lowest predicted risk that gets a call",
            f"{out['cutoff']:.0%}" if out["cutoff"] is not None else "nobody is phoned",
        )
        st.markdown(how)

    # Headline cost of the chosen setting; the breakdown is further down the tab
    if has_costs:
        with st.container(border=True):
            st.markdown(
                f"**What this setting costs and might save, per week** (about {weekly_calls:.0f} calls, "
                "for the upcoming appointments in your filters)"
            )
            # Fixed columns rather than a horizontal container, so the four cards never wrap
            # three and one. Keyed so style.css shows them two per line on narrower screens.
            c1, c2, c3, c4 = st.container(key="reflow_halves_costs").columns(4, gap="small")
            c1.metric(
                "Cost of calls",
                f"£{econ['call_cost']:,.0f}",
                border=True,
                height="stretch",
                icon=":material/call:",
                help=f"{weekly_calls:.0f} calls at £{cost_per_call:.2f} each: "
                f"{econ['staff_hours']:.1f} hours of staff time.",
            )
            c2.metric(
                "Pays for itself at",
                f"{be:.1%}" if be <= 1 else "more than everyone",
                border=True,
                height="stretch",
                icon=":material/balance:",
                help="The share of the people the calls reach, who would have missed, that the calls "
                "must get to attend to cover their staff time. It doesn't depend on your guess.",
            )
            c3.metric(
                "Possibly saved",
                f"£{econ['value_avoided']:,.0f}",
                # Sign first, so st.metric reads a loss as negative
                f"{'-' if econ['net'] < 0 else '+'}£{abs(econ['net']):,.0f} after the cost of calls",
                border=True,
                height="stretch",
                icon=":material/savings:",
                help=f"If a call gets {effect}% of the people it reaches, who would have missed, "
                "to attend.",
            )
            c4.metric(
                "Missed appointments avoided",
                f"{weekly_avoided:.1f}" if weekly_avoided < 10 else f"{weekly_avoided:.0f}",
                border=True,
                height="stretch",
                icon=":material/event_available:",
                help=f"About {weekly_reached:.0f} calls a week would reach someone who would otherwise "
                f"miss their appointment. If {effect}% of them attend instead, that's about "
                f"{weekly_avoided:.1f} fewer missed appointments a week: the assumption behind "
                "'Possibly saved'.",
            )
            st.markdown(
                (
                    f"Your guess of {effect}% is {'above' if effect / 100 > be else 'below'} the "
                    f"break-even, so the calls would {'more than ' if effect / 100 > be else 'not '}"
                    "cover their cost."
                    if be <= 1
                    else "Even if every call got the patient to attend, the calls would cost more than "
                    "the missed appointments they reach."
                )
                + " Breakdown and caveats below."
            )
    elif out["calls"] > 0:
        st.info(
            "None of the upcoming appointments in your filters would get a call with this setting, so "
            "there is nothing to cost. Widen the sidebar filters or lower the bar."
        )

    if out["calls"] == 0:
        st.info("With this setting nobody would get a call. Raise the number of calls or lower the bar.")
    else:
        # Keyed so style.css puts the cards under the chart on narrower screens
        left, right = st.container(key="reflow_stack_icons").columns([0.62, 0.38], gap="large")
        with left:
            st.plotly_chart(charts.icon_array(out), width="stretch", config={"displayModeBar": False})
            st.caption(
                "Each marker is one of 1,000 past appointments. \"Would get a call\" is the model's list "
                "(no calls were made). \"In reality\" is whether the patient really attended."
            )
            with st.expander("Show these counts as a table"):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Group": [label for _, label in calls.GROUPS],
                            "Per 1,000 appointments": [out[k] for k, _ in calls.GROUPS],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
        with right:
            st.markdown(f"**If we phoned these {out['calls']} appointments in every 1,000:**")
            # Short titles on the cards, with the explanation in a caption under each number
            with st.container(border=True, gap="small"):
                st.metric(
                    "Precision",
                    f"{out['precision']:.0%}",
                    f"{(out['precision'] - out['random_precision']) * 100:+.0f} pts vs random "
                    f"({out['random_precision']:.0%})",
                    icon=":material/call:",
                )
                st.caption(
                    f"{out['called_dna']} of the {out['calls']} calls would reach someone who then did not attend."
                )
            with st.container(border=True, gap="small"):
                st.metric("Recall", f"{out['recall']:.0%}", icon=":material/person_search:")
                st.caption(
                    f"We would reach {out['called_dna']} of the {out['dna_total']} patients who did not attend. "
                    f"The other {out['not_called_dna']} would get no call."
                )
            with st.container(border=True, gap="small"):
                st.metric("Came anyway", f"{out['called_attended']}", icon=":material/event_available:")
                st.caption(
                    f"Calls that would go to people who attended: "
                    f"{out['called_attended'] / out['calls']:.0%} of the {out['calls']} calls."
                )
            st.caption(
                f"Counts are scaled from {meta['n_past_test']:,} past appointments to 1,000, so they can be out "
                "by a few. They count appointments that were missed without a call, not missed appointments a "
                "call would have prevented."
            )

        st.divider()
        st.subheader("What difference might the calls make?")
        st.markdown(
            "Reaching someone who would have missed their appointment is not the same as getting them to "
            "come. **Nobody knows yet how many a call would change**, so it is an assumption: set it with "
            f"the slider at the top of the tab (now **{effect}%**)."
        )
        avoided = out["called_dna"] * effect / 100

        m1, m2, m3 = st.container(key="reflow_docked_difference").columns(3, gap="large")
        m1.metric(
            "Possibly avoided, per 1,000",
            f"{avoided:.0f}",
            help="Missed appointments a call might avoid, per 1,000 past appointments, if the assumption above holds.",
        )
        m2.metric(
            "Calls for each one avoided",
            f"{out['calls'] / avoided:.0f}" if avoided else "no effect assumed",
        )
        if avoided:
            m2.caption(
                f"About **£{out['calls'] / avoided * cost_per_call:,.0f}** of staff time for each "
                f"one avoided, at £{cost_per_call:.2f} a call."
            )
        m3.metric(
            "Possibly avoided, per week",
            f"{weekly_avoided:.0f}",
            help="For the upcoming appointments in the sidebar filters, if the assumption above holds.",
        )

        st.markdown(
            f"**For the upcoming appointments in your filters** ({len(filtered_data):,} over {n_weeks} "
            f"week{'s' if n_weeks != 1 else ''}), this setting means about **{weekly_calls:.0f} calls a week**. "
            f"Going by the past appointments, about **{weekly_reached:.0f}** of them would reach someone who "
            f"would otherwise miss their appointment, and if a call gets {effect}% of them to come, about "
            f"**{weekly_avoided:.0f}** fewer missed appointments a week."
        )
        st.caption(
            "Only this paragraph and the last figure above follow the sidebar filters, including the risk and "
            "confidence filters: filter to high-risk appointments only and there are fewer to phone. The chart "
            "uses all past appointments. Weekly figures use the past precision, so they are rough."
        )

        st.divider()
        st.subheader("How the cost is worked out")
        if not has_costs:
            st.info(
                "None of the upcoming appointments in your filters would get a call with this "
                "setting, so there is nothing to cost. Widen the sidebar filters or lower the bar."
            )
        else:
            st.markdown(
                f"Per week, for the upcoming appointments in your filters. A call costs about "
                f"**£{cost_per_call:.2f}** of staff time, and a missed appointment among the patients "
                f"these calls would reach costs about **£{dna_cost:,.0f}** on average (more for new "
                "appointments and some specialties). Change these under **Cost assumptions** in the "
                "sidebar. The break-even at the top is the cost of calls divided by the missed "
                "appointments reached."
            )
            with st.container(horizontal=True):
                st.metric(
                    "Staff time",
                    f"{econ['staff_hours']:.1f} hours",
                    border=True,
                    icon=":material/schedule:",
                    help=f"{weekly_calls:.0f} calls at {minutes_per_call:g} minutes each.",
                )
                st.metric(
                    "Cost of calls",
                    f"£{econ['call_cost']:,.0f}",
                    border=True,
                    icon=":material/call:",
                    help=f"{weekly_calls:.0f} calls at £{cost_per_call:.2f} each.",
                )
                st.metric(
                    "Missed appointments reached",
                    f"£{econ['value_reached']:,.0f}",
                    border=True,
                    icon=":material/person_search:",
                    help=f"About {weekly_reached:.0f} calls a week would reach someone who would "
                    f"otherwise miss their appointment, at about £{dna_cost:,.0f} each.",
                )
                st.metric(
                    "Possibly saved",
                    f"£{econ['value_avoided']:,.0f}",
                    # Sign first, so st.metric reads a loss as negative
                    f"{'-' if econ['net'] < 0 else '+'}£{abs(econ['net']):,.0f} after the cost of calls",
                    border=True,
                    icon=":material/savings:",
                    help=f"The missed appointments reached, times your guess that a call gets {effect}% "
                    "of them to attend.",
                )
            st.caption(
                "These are estimates from the assumptions in the sidebar. A missed appointment's cost "
                "is the value of a clinic slot, not cash that comes back: the clinic still runs, and the "
                "gain is slots used and shorter waits. Prices are not costs. The call cost counts staff "
                "time only, not phones, supervision or rebooking."
            )

    st.warning(
        "**We don't know that a call will change whether someone attends.** The chart shows who the calls "
        "would *reach*, not missed appointments they would *prevent*.\n\n"
        "- Some people would still miss their appointment after a call, and some would have come anyway.\n"
        "- The model picks the people most like those who missed appointments before, not the people a call "
        "would help most. Someone at high risk may be the hardest to help with a phone call.\n"
        "- Calls to people who would have come anyway still take staff time, and could worry them.\n"
        "- The effect slider is a guess. The only way to know is to test it: for example, randomly phone half "
        "of the flagged patients and compare how many from each half attend.",
        icon=":material/warning:",
    )
