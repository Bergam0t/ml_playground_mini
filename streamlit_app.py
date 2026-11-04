import base64
import hashlib
import json
from pathlib import Path

import streamlit as st
import pandas as pd

import calls
import charts
import explain
import weekly
from demo_data import DATA_DIR, load_forecast  # DEMO: synthetic data replaces SQL

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


def DNA_Risk(prob):
    if prob < 0.15:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.80:
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
        help="Model forecast: will the patient miss this appointment? Red Yes, green No.",
    ),
    "Confidence": st.column_config.TextColumn(
        width=105,
        help="Model forecast: how sure the model is. Darker teal is less sure.",
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


def High_Risk_DNA(prob, conf):
    if prob > 0.8 and conf > 0.8:
        return 1
    return 0


def demo_notice():
    st.warning(
        "DEMONSTRATION VERSION - all data shown is synthetic. "
        "No real patients or Trust data are displayed."
    )


# Streamlit app
# The tabs come first so style.css can pin them to the top as a navbar; the app title
# is drawn at the left of that bar by style.css.
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

condition = (data["DNA_Prob_Forecast"] > 0.7) & (
    data["Confidence_Score"] > 0.7
)
data["High_Risk_DNA"] = condition.astype(int)

# data['High_Risk_DNA'] = data['DNA_Prob_Forecast','Confidence_Score'].apply(High_Risk_DNA)
data["Probabilty"] = data["DNA_Prob_Forecast"] * 100


# Filters
st.sidebar.header("Filters")
# Start with the full data
filtered_data = data.copy()

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

# DNA Risk options (from filtered data)
risk_options = ["Low", "Medium", "High", "Very High"]
available_risks = [
    r for r in risk_options if r in filtered_data["Risk"].unique()
]
selected_dna_risk = st.sidebar.selectbox(
    "Select DNA Risk", options=["All"] + available_risks
)

if selected_dna_risk != "All":
    filtered_data = filtered_data[filtered_data["Risk"] == selected_dna_risk]


# --- NEW: Filter for Confidence Level ---
confidence_options = ["Low", "Medium", "High", "Very High"]
available_confidences = [
    c for c in confidence_options if c in filtered_data["Confidence"].unique()
]
selected_confidence = st.sidebar.selectbox(
    "Select Confidence Level", options=["All"] + available_confidences
)

if selected_confidence != "All":
    filtered_data = filtered_data[
        filtered_data["Confidence"] == selected_confidence
    ]
# --- END of new filter ---

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
if filtered_data.empty:
    st.info("No appointments match the selected filters.")
    st.stop()

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
        )
        st.metric(
            "DNA rate (high confidence & high risk)",
            f"{filtered_data['High_Risk_DNA'].mean() * 100:.1f}%",
            icon=":material/warning:",
            border=True,
        )

    st.info(
        "**Click a row in the table** to see what made the model give that "
        "appointment its risk of not attending. The coloured columns are the model's forecast.",
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
            example_col, steps_col = st.columns([4, 5], vertical_alignment="center")
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

# ---------------------------------------------------------------- tab 2
with tab_weekly:
    demo_notice()

    st.subheader("Weekly Forecast Summary")

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
        "The model gives each appointment a **risk** of not attending, not a yes or no. If the team "
        "phones patients beforehand, someone has to choose who gets a call. You can choose by **how many "
        "calls the team can make**, or by **how high the predicted risk and confidence must be**."
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
        ["Calls the team can make", "Predicted risk and confidence"],
        default="Calls the team can make",
        key="call_mode",
    ) or "Calls the team can make"

    if mode == "Calls the team can make":
        # Slider runs up to the busiest filtered week, rounded up to a multiple of 10. Clamp
        # the kept value first, so changing the filters never leaves it out of range.
        busiest = max(10, int(-(-per_week.max() // 10) * 10))
        if st.session_state.get("calls_per_week", 0) > busiest:
            st.session_state["calls_per_week"] = busiest
        st.session_state.setdefault("calls_per_week", min(100, busiest))
        cap = st.slider(
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
        c1, c2 = st.columns(2, gap="large")
        with c1:
            use_risk = st.toggle("Use predicted risk", value=True, key="use_risk")
            risk_min = st.slider(
                "Phone if the predicted risk is at least",
                0,
                100,
                70,
                step=5,
                format="%d%%",
                key="risk_min",
                disabled=not use_risk,
            )
        with c2:
            use_conf = st.toggle("Use confidence", value=True, key="use_conf")
            conf_min = st.slider(
                "...and the confidence is at least",
                50,
                99,
                70,
                step=1,
                format="%d%%",
                key="conf_min",
                disabled=not use_conf,
            )
        risk_min = risk_min / 100 if use_risk else None
        conf_min = conf_min / 100 if use_conf else None
        called = calls.called_by_threshold(p_past, past["Confidence_Score"], risk_min, conf_min)
        upcoming_called = pd.Series(
            calls.called_by_threshold(
                filtered_data["DNA_Prob_Forecast"], filtered_data["Confidence_Score"], risk_min, conf_min
            ),
            index=filtered_data.index,
        )
        how = (
            "**Here you set the bar, and the number of calls follows from it.** A lower bar means more calls, "
            "and more of them are to people who would have come anyway. 70% and 70% match the app's "
            "existing high-risk flag."
        )

    out = calls.call_outcomes(called, past["Did_Not_Attend"], p_past)

    with st.container(border=True):
        k1, k2 = st.columns([0.35, 0.65], gap="large")
        k1.metric("Calls per 1,000 appointments", f"{out['calls']}")
        k2.metric(
            "So we phone anyone with a predicted risk of at least"
            if mode == "Calls the team can make"
            else "Lowest predicted risk that gets a call",
            f"{out['cutoff']:.0%}" if out["cutoff"] is not None else "nobody is phoned",
        )
        st.markdown(how)

    if out["calls"] == 0:
        st.info("With this setting nobody would get a call. Raise the number of calls or lower the bar.")
    else:
        left, right = st.columns([0.62, 0.38], gap="large")
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
            "come. **Nobody knows yet how many a call would change**, so this is an assumption you can vary."
        )
        effect = st.slider(
            "Suppose a call gets this share of the people it reaches, who would have missed, to attend instead",
            0,
            50,
            20,
            step=5,
            format="%d%%",
            key="call_effect",
            help="An assumption, not a measurement. Try a few values: 0% means the calls make no difference.",
        )
        avoided = out["called_dna"] * effect / 100

        weekly_calls = upcoming_called.sum() / n_weeks
        weekly_reached = weekly_calls * out["precision"]
        weekly_avoided = weekly_reached * effect / 100

        m1, m2, m3 = st.columns(3, gap="large")
        m1.metric(
            "Possibly avoided, per 1,000",
            f"{avoided:.0f}",
            help="Missed appointments a call might avoid, per 1,000 past appointments, if the assumption above holds.",
        )
        m2.metric(
            "Calls for each one avoided",
            f"{out['calls'] / avoided:.0f}" if avoided else "no effect assumed",
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
