import base64
import hashlib
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd

import calls
import charts
import costs
import explain
import rule_game
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


# The "Your rule vs the model" tab: past appointments with a score per clue, and the
# model's results as it learns (both precomputed by build/rule_game_data.py)
@st.cache_data
def load_rule_game():
    return rule_game.load_past(), rule_game.load_learning()


# What every rule the visitor can make would catch, for "Your rule ranks #N of 511"
@st.cache_data
def load_rule_catches():
    return rule_game.all_rule_catches(rule_game.load_past())


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
# The assumed share of the people a call reaches, who would have missed, that it gets to
# attend (%): the starting guess on the phone tab and in the savings on the appointments tab
DEFAULT_CALL_EFFECT = 50


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
    "Low": "#1c3436",
    "Medium": "#164f53",
    "High": "#0f6a70",
    "Very High": "#07858d",
}
RISK_COLOURS = {
    "Low": "#1e5631",
    "Medium": "#7a6a12",
    "High": "#9a4a12",
    "Very High": "#8b1e2d",
}


def forecast_cell_styles(df):
    """CSS for each cell of the appointments table: only the forecast columns are coloured."""

    def css(colour):
        return f"background-color: {colour}; color: #ffffff;"

    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles["Expected DNA"] = df["Expected DNA"].map(EXPECTED_DNA_COLOURS).map(css)
    styles["Confidence"] = df["Confidence"].map(CONFIDENCE_COLOURS).map(css)
    # Same bands as the sidebar's "DNA Risk" filter
    styles["Probabilty"] = (
        (df["Probabilty"] / 100).map(DNA_Risk).map(RISK_COLOURS).map(css)
    )
    return styles


# Appointment tables (the all-appointments view, and a day's list in the next-week summary):
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

# Fixed column widths (about 1,110px in total; about 1,300px on the call list, with its two
# extra columns) so the whole table fits a 1,400px main area; the grid stretches them to fill
# any space left over.
APPOINTMENT_COLUMNS = {
    "Appointment Date": st.column_config.DateColumn(
        "Appointment Date", format="DD/MM/YYYY", width=110
    ),
    "Time": st.column_config.TextColumn(width=65),
    "Patient ID": st.column_config.TextColumn(width=95),
    "Patient Name": st.column_config.TextColumn(width=150),
    "Mobile Number": st.column_config.TextColumn(width=120),
    #'Mobile Number (GP Record)': st.column_config.TextColumn(),
    "Specialty": st.column_config.TextColumn(width=145),
    "Location": st.column_config.TextColumn(width=155),
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


# The call list only: the admin team needs a name and a time to phone and remind. Not shown in
# the all-appointments view
CALL_LIST_FIELDS = {
    "Appointment_Time": "Time",
    "Patient Name": "Patient Name",
}


def appointment_table(df, by_patient=False, call_list=False):
    """Appointments to show in a table, with the forecast columns coloured.

    `by_patient` puts the patient columns first and shades every other patient's rows,
    for a table sorted so each patient's appointments are together. `call_list` adds the
    appointment time and the patient's name.
    """
    fields = list(APPOINTMENT_FIELDS)
    if call_list:
        fields.insert(1, "Appointment_Time")
        fields.insert(3, "Patient Name")
    if by_patient:
        patient = ["MRN", "Mobile Telephone"]
        fields = patient + [f for f in fields if f not in patient]
    shown = df[fields].rename(columns={**APPOINTMENT_FIELDS, **CALL_LIST_FIELDS})
    styler = shown.style.apply(forecast_cell_styles, axis=None)
    return styler.apply(patient_bands, axis=None) if by_patient else styler


def patient_bands(df):
    """Shade every other patient's rows (outside the coloured forecast columns)."""
    patient_no = (df["Patient ID"] != df["Patient ID"].shift()).cumsum()
    plain = [
        c for c in df.columns if c not in ("Expected DNA", "Confidence", "Probabilty")
    ]
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles.loc[patient_no % 2 == 0, plain] = "background-color: #2e2c2c;"
    return styles


def demo_notice_full():
    """The fuller version of the notice, beside tab 1's opening text."""
    st.warning(
        "**DEMONSTRATION VERSION**\n\n"
        "This is a demonstration of a :blue[**real tool**], currently being piloted by "
        "outpatient teams in a London trust, which predicts attendance for real upcoming appointments.\n\n"
        "Here, all the data for training and display is synthetic: "
        "no real patients or Trust data are displayed.\n\n"
        "Read the story behind this HSMA project in this "
        "[impact poster](https://hsma.co.uk/impact_posters/Outpatient%20non%20attendance.png), "
        "or hear more directly from the associate "
        "[here](https://youtu.be/tWcLIMTQb8A).",
        icon=":material/science:",
    )


def intro_columns(key, notice=demo_notice_full, widths=(0.65, 0.35)):
    """Columns for a tab's opening text (left) beside the demonstration notice (right).

    Returns the left column. The reflow_stack_ key stacks them on narrower screens (style.css).
    """
    left, right = st.container(key=f"reflow_stack_{key}").columns(
        list(widths), gap="large"
    )
    with right:
        notice()
    return left


def go_to_tab(label):
    """Switch tab from a button; the new tab would otherwise open at the old scroll position.
    Ported from the geographic playground mini."""
    st.session_state.page = label
    st.session_state.scroll_to_top = True


# Everything the Your rule vs the model tab remembers about a visitor's go
GAME_KEYS = ("game_clues", "game_best", "game_last", "game_prev", "game_trained")


def reset_game():
    """Clear the game for the next person at the stand."""
    for key in GAME_KEYS:
        st.session_state.pop(key, None)
    st.session_state.scroll_to_top = True


def go_to_call_list():
    """From the bottom of the all-appointments view to the admin team's call list view."""
    st.session_state.appt_view = VIEW_WEEK
    st.session_state.scroll_to_top = True


def go_to_forecast_calls(risk_min):
    """From the predicted savings on the appointments tab to the phone tab, set to phone
    everyone at or above `risk_min` with the default guess at what a call changes, so the
    figures match."""
    st.session_state.call_mode = "Predicted risk"
    st.session_state.risk_min = round(risk_min * 100)
    st.session_state.call_effect = DEFAULT_CALL_EFFECT
    go_to_tab(TAB_LABELS[2])


# Streamlit app
# The tabs come first so style.css can pin them to the top as a navbar; the app title
# is drawn at the left of that bar by style.css.
# Dropdown in place of the tab pills on narrower screens (shown by style.css)
add_mobile_nav()
# Keyed and rerun on change, so buttons can switch tab (go_to_tab), as in the geographic mini
TAB_LABELS = [
    "Your rule vs the model",
    "Upcoming appointments",
    "How many people should we phone?",
]
tab_game, tab_appts, tab_calls = st.tabs(TAB_LABELS, key="page", on_change="rerun")
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

# Which view the Upcoming appointments tab shows. Only offered on that tab; the tabs are
# keyed, so the page state says which one is open.
VIEW_ALL = "Show all appointments"
VIEW_WEEK = "Show next week's call list"
VIEW_ICONS = {VIEW_ALL: ":material/table_rows:", VIEW_WEEK: ":material/call:"}
DEFAULT_MEASURE = next(iter(weekly.MEASURES))  # the first: the strictest list
if st.session_state.get("page") == TAB_LABELS[1]:
    st.sidebar.segmented_control(
        "Click the buttons below to change the view",
        [VIEW_ALL, VIEW_WEEK],
        key="appt_view",
        format_func=lambda v: f"{VIEW_ICONS[v]} {v}",
    )
view = st.session_state.get("appt_view") or VIEW_ALL

# Filters. Hidden on the How many people should we phone? tab, which uses all the upcoming
# appointments and only shows the cost assumptions. The choices made on the other tabs are
# kept, and still applied to them, while it is open.
show_filters = st.session_state.get("page") != TAB_LABELS[2]


def filter_select(label, options, key, **kwargs):
    """A sidebar filter; while the filters are hidden, the choice kept from the other tabs."""
    if show_filters:
        return st.sidebar.selectbox(label, options=options, key=key, **kwargs)
    kept = st.session_state.get(key, "All")
    if kept not in options:
        kept = "All"
    # Streamlit forgets a widget's value on a run that doesn't draw it, unless it is set again
    st.session_state[key] = kept
    return kept


if show_filters:
    st.sidebar.header("Filters")
    st.sidebar.caption("Choose which upcoming appointments the app shows.")
# Start with the full data
filtered_data = data.copy()

# Facts about the appointment first, then the model's forecast for it
if show_filters:
    st.sidebar.subheader("Appointment details")

# Allow the user to filter by Site
reporting_site_options = sorted(filtered_data["Reporting_Site"].dropna().unique())
selected_reporting_site = filter_select(
    "Select Reporting Site", ["All"] + list(reporting_site_options), "filter_site"
)

if selected_reporting_site != "All":
    filtered_data = filtered_data[
        filtered_data["Reporting_Site"] == selected_reporting_site
    ]


# Specialty filter options (always from full data)
specialty_options = sorted(filtered_data["Specialty"].dropna().unique())
selected_specialty = filter_select(
    "Select Specialty", ["All"] + list(specialty_options), "filter_specialty"
)

if selected_specialty != "All":
    filtered_data = filtered_data[filtered_data["Specialty"] == selected_specialty]

# Appointment Type options (from filtered data)
appointment_type_options = sorted(filtered_data["Appointment Type"].dropna().unique())
selected_appointment_type = filter_select(
    "Select Appointment Type", ["All"] + list(appointment_type_options), "filter_type"
)

if selected_appointment_type != "All":
    filtered_data = filtered_data[
        filtered_data["Appointment Type"] == selected_appointment_type
    ]

# Appointment Location options (from filtered data)
appointment_location_options = sorted(
    filtered_data["Appointment Location Text"].dropna().unique()
)
selected_appointment_location = filter_select(
    "Select Appointment Location",
    ["All"] + list(appointment_location_options),
    "filter_location",
)

if selected_appointment_location != "All":
    filtered_data = filtered_data[
        filtered_data["Appointment Location Text"] == selected_appointment_location
    ]

if filtered_data.empty:  # DEMO: avoid crashing on empty filter combinations
    st.info("No appointments match the selected filters.")
    st.stop()

if show_filters:
    st.sidebar.divider()
    st.sidebar.subheader("Model forecast")

# DNA Risk options (from filtered data)
risk_options = ["Low", "Medium", "High", "Very High"]
available_risks = [r for r in risk_options if r in filtered_data["Risk"].unique()]
selected_dna_risk = filter_select(
    "Select DNA Risk",
    ["All"] + available_risks,
    "filter_risk",
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
selected_confidence = filter_select(
    "Select Confidence Level",
    ["All"] + available_confidences,
    "filter_confidence",
    # Bands from Confidence_Score_Txt()
    help="How clear-cut the model's yes-or-no call is: how far the risk is from 20%, where the "
    "forecast changes from No to Yes. It follows from the risk.  \n"
    "**Low:** under 50% (risk 8% to 43%)  \n"
    "**Medium:** 50% to 80% (risk 3% to 8% or 43% to 69%)  \n"
    "**High:** 80% to 90% (risk 1.3% to 3% or 69% to 83%)  \n"
    "**Very High:** 90% or more (risk under 1.3% or over 83%)",
)

if selected_confidence != "All":
    filtered_data = filtered_data[filtered_data["Confidence"] == selected_confidence]
# --- END of new filter ---

# Cost assumptions: not filters, so they stay out of filter_key below and editing them
# never clears a table selection
if show_filters:
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

    share_lost = (
        st.slider(
            "Share of a missed appointment's cost really lost",
            0,
            100,
            100,
            step=5,
            format="%d%%",
            key="cost_share_lost",
            help="Lower this if some empty slots get re-used, for example by overbooking or a "
            "clinician catching up on other work.",
        )
        / 100
    )
    # A pair of boxes per specialty rather than a table, which is cramped at the sidebar's
    # width. They make the same table as costs.dna_cost_table(), for appointment_costs().
    st.markdown("**Cost of a missed appointment (£)**")
    st.caption(
        "Illustrative values, loosely in line with NHS Payment Scheme outpatient prices: "
        "replace them with your Trust's own costs."
    )
    cost_rows = []
    for i, (specialty, new_default, fu_default) in enumerate(
        costs.dna_cost_table().itertuples(index=False)
    ):
        with st.container(key=f"cost_dna_{i}", gap="small"):
            st.markdown(f"**{specialty}**")
            new_col, fu_col = st.columns(2, gap="small")
            new_cost = new_col.number_input(
                "New (£)",
                min_value=0,
                value=int(new_default),
                step=10,
                key=f"cost_dna_{i}_new",
                help=f"A missed first appointment in {specialty}.",
            )
            fu_cost = fu_col.number_input(
                "Follow up (£)",
                min_value=0,
                value=int(fu_default),
                step=10,
                key=f"cost_dna_{i}_fu",
                help=f"A missed follow-up appointment in {specialty}.",
            )
        cost_rows.append((specialty, new_cost, fu_cost))
    dna_costs = pd.DataFrame(cost_rows, columns=["Specialty", *costs.APPOINTMENT_TYPES])

if filtered_data.empty:
    st.info("No appointments match the selected filters.")
    st.stop()

filtered_data["DNA_Cost"] = costs.appointment_costs(
    filtered_data, dna_costs, share_lost
)

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
        )
    ).encode()
).hexdigest()[:10]

# The lowest predicted risk on each call list (weekly.MEASURES), for the savings box
MEASURE_RISK_MIN = {
    "High_Risk_DNA": HIGH_RISK_MIN,
    "DNA_Forecast": FORECAST_MIN,
    "DNA_Prob_Forecast": 0.0,
}


def savings_box(called, risk_min, label, caption, scope, key, none_text, weeks=None):
    """Predicted savings from phoning every appointment in `called` (all those at or above
    `risk_min`), if a call gets DEFAULT_CALL_EFFECT% of those who would have missed to attend.
    Worked out as the phone tab does: calls reach a would-be DNA as often as they did in the
    past appointments at the same cut-off. A button opens the phone tab on the same setting.
    `scope` says what the phone tab covers that this doesn't; `none_text` is shown instead
    when nobody would be phoned; `weeks` adds a weekly figure."""
    n_calls = len(called)
    dna_cost = costs.cost_per_dna_reached(called) if n_calls else None
    if dna_cost is None:
        st.caption(none_text)
        return
    past = load_past()
    precision = past.loc[past["DNA_Prob_Forecast"] >= risk_min, "Did_Not_Attend"].mean()
    effect = DEFAULT_CALL_EFFECT / 100
    reached = n_calls * precision
    saving = costs.call_economics(
        n_calls, reached, effect, cost_per_call, minutes_per_call, dna_cost
    )
    who = (
        f"those with a predicted risk of {risk_min:.0%} or more"
        if risk_min > 0
        else "all appointments"
    )
    per_week = (
        f" About £{saving['value_avoided'] / weeks:,.0f} a week."
        if weeks and weeks > 1
        else ""
    )
    def few(x):
        """One decimal place under 10, as on the phone tab"""
        return f"{x:.1f}" if x < 10 else f"{x:,.0f}"

    with st.container(border=True):
        saving_col, cta_col = st.container(key=f"reflow_stack_{key}").columns(
            2, gap="large", vertical_alignment="center"
        )
        saving_col.metric(
            label,
            f"£{saving['value_avoided']:,.0f}",
            # A minus only for a loss, so st.metric still colours it red; no plus or arrow
            f"{'-' if saving['net'] < 0 else ''}£{abs(saving['net']):,.0f} after the cost of calls",
            delta_arrow="off",
            icon=":material/savings:",
            help=f"{n_calls:,} calls at £{cost_per_call:.2f} each: £{saving['call_cost']:,.0f}, "
            f"or {few(saving['staff_hours'])} hours of staff time. In past appointments, "
            f"{precision:.0%} of {who} were missed, so about {reached:,.0f} calls would reach "
            f"someone who would otherwise miss. If {DEFAULT_CALL_EFFECT}% of them attend instead, "
            f"that's about {few(reached * effect)} fewer missed appointments, each valued at about "
            f"£{dna_cost:,.0f}.{per_week} The costs come from **Cost assumptions** in the sidebar.",
        )
        saving_col.caption(caption)
        cta_col.markdown(
            f"**{DEFAULT_CALL_EFFECT}% is a guess, not a measurement.** Nobody knows yet how "
            "many people a reminder call would change. Try other guesses, and other numbers "
            f"of calls, on the next tab. It uses all the upcoming appointments, {scope}."
        )
        cta_col.button(
            "Explore these assumptions",
            key=f"to_calls_from_{key}",
            on_click=go_to_forecast_calls,
            args=(risk_min,),
            icon=":material/arrow_forward:",
            type="primary",
        )


# ---------------------------------------------------------------- tab 1
# One tab, two views picked by the switch at the top of the sidebar
with tab_appts:
    if view == VIEW_ALL:
        # What really happened to past appointments the model did not train on
        past = load_past()
        past_dna_rate = past["Did_Not_Attend"].mean()

        def catch_rates(threshold):
            """Share of the missed appointments at or above this risk, and share of those that were missed."""
            picked = past["DNA_Prob_Forecast"] >= threshold
            missed = past["Did_Not_Attend"] == 1
            return (picked & missed).sum() / missed.sum(), missed[picked].mean()

        forecast_recall, forecast_precision = catch_rates(FORECAST_MIN)
        high_recall, high_precision = catch_rates(HIGH_RISK_MIN)

        # Opening text beside the demonstration notice (stacked on narrower screens, see style.css)
        with intro_columns("intro"):
            st.markdown(
                f"""
                ### Which appointments are patients most likely to miss?

                A machine-learning model, trained on past appointments, gives each upcoming one a
                predicted **risk of not attending** (a DNA, or 'did not attend') and a **confidence
                score**, so outpatient teams can plan where reminder calls might help most. The
                forecasts are <span class="highlight">estimates to support staff judgement, not
                decisions</span>.

                <div class="threshold-pills">
                <span class="pill">About {past_dna_rate:.0%} of past appointments were missed</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"That is the recorded outcome in {len(past):,} past appointments the model was tested on, "
                "so most predicted risks are low."
            )
            with st.expander(
                "What does the confidence score mean?", icon=":material/help:"
            ):
                st.markdown(
                    f"Confidence is how clear-cut the model's yes-or-no call is: how far the risk is "
                    f"from {FORECAST_MIN:.0%}, where the forecast changes from No to Yes. A "
                    f"{FORECAST_MIN:.0%} risk gives 0% confidence, because the model could go either "
                    "way. A 70% risk gives about 80% confidence that the patient won't attend, and a 2% "
                    "risk about 85% confidence that they will. So it is **Low for any risk between 8% "
                    "and 43%**, and rises towards both ends."
                )

        # Summary of the filtered appointments. Every card is a bordered container holding a
        # metric and a one-line caption, so the four are the same height; the 20%+ and 40%+
        # pills sit above two of them, so the row is bottom-aligned to keep the boxes level.
        n_filtered = len(filtered_data)

        def pill(risk_band, threshold):
            st.markdown(
                f'<span class="pill" style="background-color:{RISK_COLOURS[risk_band]};color:#fff">'
                f"{threshold:.0%}+ risk</span>",
                unsafe_allow_html=True,
            )

        def share_of_appointments(count):
            return f"{count / n_filtered:.1%} of upcoming appointments"

        with st.container(horizontal=True, vertical_alignment="bottom"):
            # with st.container(border=True):
            #     st.metric(
            #         "Appointments (filtered)",
            #         f"{n_filtered:,}",
            #         icon=":material/calendar_month:",
            #     )
            #     st.caption("Matching the filters on the left")
            with st.container():
                pill("High", FORECAST_MIN)
                with st.container(border=True):
                    n_forecast = int(filtered_data["DNA_Forecast"].sum())
                    st.metric(
                        "All forecast DNAs",
                        f"{n_forecast:,}",
                        share_of_appointments(n_forecast),
                        delta_color="off",
                        delta_arrow="off",
                        icon=":material/event_busy:",
                        help=f"Appointments the model forecasts as a DNA: a predicted risk of "
                        f"{FORECAST_MIN:.0%} or more (the 'Expected DNA' column). Not all of them "
                        "will be missed, and some lower-risk appointments will be.",
                    )
                    st.caption(
                        f"Catches {forecast_recall:.0%} of past DNAs",
                        help=f"Of the {int(past['Did_Not_Attend'].sum()):,} appointments that were "
                        f"missed in the past appointments the model was tested on, "
                        f"{forecast_recall:.0%} had a predicted risk of {FORECAST_MIN:.0%} or more. "
                        f"Of all those forecast as a DNA, {forecast_precision:.0%} were missed.",
                    )
            with st.container():
                pill("Very High", HIGH_RISK_MIN)
                with st.container(border=True):
                    n_high = int(filtered_data["High_Risk_DNA"].sum())
                    st.metric(
                        "DNAs we're confident about",
                        f"{n_high:,}",
                        share_of_appointments(n_high),
                        delta_color="off",
                        delta_arrow="off",
                        icon=":material/warning:",
                        help=f"Appointments with a predicted risk of {HIGH_RISK_MIN:.0%} or more "
                        "(the Very High band).",
                    )
                    st.caption(
                        f"Catches {high_recall:.0%} of past DNAs",
                        help=f"Of the {int(past['Did_Not_Attend'].sum()):,} appointments that were "
                        f"missed in the past appointments the model was tested on, "
                        f"{high_recall:.0%} had a predicted risk of {HIGH_RISK_MIN:.0%} or more. "
                        f"Of all those flagged, {high_precision:.0%} were missed.",
                    )
            with st.container(border=True):
                st.metric(
                    "Expected cost of missed appointments",
                    f"£{costs.expected_dna_cost(filtered_data):,.0f}",
                    icon=":material/payments:",
                    help="Each appointment's predicted risk of not attending times the cost of "
                    "missing it, added up. The costs are assumptions you can change under "
                    "**Cost assumptions** in the sidebar.",
                )
                dates = pd.to_datetime(filtered_data["Activity_Date"])
                days = (dates.max() - dates.min()).days + 1
                weeks = -(
                    -days // 7
                )  # whole weeks, rounded up (appointments run Monday to Friday)
                length = (
                    f"{days} day{'s' * (days != 1)}" if days <= 7 else f"{weeks} weeks"
                )
                st.caption(
                    f"Appointments {dates.min():%d %b} to {dates.max():%d %b} ({length})"
                )

        # Predicted savings from phoning every forecast DNA in the period above
        savings_box(
            filtered_data[filtered_data["DNA_Forecast"] == 1],
            FORECAST_MIN,
            f"Predicted savings over the next {length} if we phone every forecast DNA",
            f"If half of those who would have missed attend instead, with {n_forecast:,} calls "
            f"between {dates.min():%d %b} and {dates.max():%d %b}.",
            "not just your filters",
            "savings",
            "None of the appointments in your filters is forecast as a DNA, so there are no "
            "calls to cost.",
            weeks=weeks,
        )

        st.subheader("Predicted Non-Attender Summary - Weekly")
        # Only the cut-off lists: "ranked by risk" describes a call list, not a colouring
        grid_measures = {n: mm for n, mm in weekly.MEASURES.items() if mm.flag}
        measure = (
            st.segmented_control(
                "Colour the grid by",
                list(grid_measures),
                default=DEFAULT_MEASURE,
                key="grid_measure",
                help="  \n".join(
                    f"**{name}:** {mm.help}" for name, mm in grid_measures.items()
                ),
            )
            or DEFAULT_MEASURE
        )
        m = weekly.MEASURES[measure]
        column = m.column
        h = weekly.headline(filtered_data, column)
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
        cell_styles = rates.map(lambda r: weekly.cell_style(r, low, high)).set_axis(
            week_index
        )

        st.caption(
            f"Each cell: appointments · {m.cell_label} "
            f"({'rate' if m.flag else 'average predicted risk'}). "
            f"Green below {low:.0%}, red above {high:.0%}."
        )
        st.info(
            "**Click a day in the table** to list just that day's appointments below. Click it again "
            "to show them all.",
            icon=":material/touch_app:",
        )
        week_table = st.dataframe(
            cell_text.style.apply(lambda _: cell_styles, axis=None),
            width="stretch",
            on_select="rerun",
            selection_mode="single-cell",
            key=f"weekly_{filter_key}",
        )

        # The clicked day narrows the appointment list (clicks on the week labels are ignored)
        picked = [
            (r, day)
            for r, day in week_table.selection.cells
            if r < len(counts) and day in counts.columns
        ]
        day_key = "all"
        table_data = filtered_data
        if picked:
            r, day = picked[0]
            date = counts.index[r] - pd.Timedelta(days=6 - weekly.DAYS.index(day))
            table_data = filtered_data[
                pd.to_datetime(filtered_data["Activity_Date"]) == date
            ]
            day_key = f"{date:%Y%m%d}"
            st.subheader(
                f"{date:%A} {date.day} {date:%B}: {len(table_data)} appointments"
            )

        st.subheader("Appointment Details")

        st.info(
            "**Click a row in the table** (highest risk first) to see what made the model give that "
            "appointment its risk of not attending. Until you do, the chart below shows what the "
            "model relies on most. The coloured columns are the model's forecast.",
            icon=":material/touch_app:",
        )

        # Highest risk first, so the first rows are the red ones worth clicking. Sorted here
        # rather than in the grid, so the selected position still matches table_data.
        table_data = table_data.sort_values(
            "DNA_Prob_Forecast", ascending=False, kind="stable"
        )
        table = st.dataframe(
            appointment_table(table_data),
            column_config=APPOINTMENT_COLUMNS,
            hide_index=True,
            alt="Upcoming appointments with the forecast risk of not attending",
            on_select="rerun",
            selection_mode="single-row",
            key=f"appointments_{filter_key}_{day_key}",
        )

        # Selected positions refer to the rows as passed in, whatever sorting the user applies
        selected_rows = [r for r in table.selection.rows if r < len(table_data)]
        if selected_rows:
            row = table_data.iloc[selected_rows[0]]
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
                "How to read this chart",
                expanded=True,
                key="chart_guide",
                icon=":material/help:",
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
                f"perfectly accurate: tested on past appointments it hadn't seen, it gave a missed "
                f"appointment a higher risk than an attended one about {meta['auc_test'] * 100:.0f} "
                f"times in 100."
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
            forecast_filtered = (
                selected_dna_risk != "All" or selected_confidence != "All"
            )
            show_filtered = (
                is_filtered and n >= MIN_GROUP_FOR_CHART and not forecast_filtered
            )

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
                    explain.group_importance(shap_all.loc[filtered_data["row_id"]])
                    if show_filtered
                    else None,
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
                "The model is reasonably but not perfectly accurate: tested on past appointments it "
                "hadn't seen, it gave a missed appointment a higher risk than an attended one about "
                f"{meta['auc_test'] * 100:.0f} times in 100."
            )

        # ---- The way on: a split button, styled like the game tab's (style.css)
        st.divider()
        with st.container(key="appts_next"):
            call_list, plan = st.container(key="reflow_stack_appts_next").columns(
                2, gap="large"
            )
            call_list.markdown(
                "### See the list for the admin team\n"
                "What the team would actually work from: next week's appointments, highest risk first, "
                "ready to phone from the top."
            )
            call_list.button(
                "See next week's call list",
                key="to_call_list",
                on_click=go_to_call_list,
                icon=":material/arrow_forward:",
                type="primary",
            )
            plan.markdown(
                "### Or plan the calls\n"
                "How many calls should the team make, and would they be worth the staff time?"
            )
            plan.button(
                "Plan the calls",
                key="to_calls_from_appts",
                on_click=go_to_tab,
                args=(TAB_LABELS[2],),
                icon=":material/arrow_forward:",
            )

    else:
        with intro_columns("intro_weekly", widths=(0.70, 0.3)):
            st.title("Prioritised call list for admin team")

            st.info(
                "**Where to draw the line is a human decision.** "
                "\n\nThe model gives every appointment a risk "
                "of not attending, not a yes or no. "
                ""
                "\n\n- A higher cut-off gives a shorter list that is "
                "more likely to reach people who really would have missed their appointment"
                "\n- A lower one catches "
                "more missed appointments but means more calls, many to people who would have come anyway. "
                "\n\nThe right balance depends on how many calls the team can make."
                "Choose a starting point above, then head to the **How many people should we phone?** tab to explore the trade-off.",
                icon=":material/balance:",
            )

        measure = (
            st.segmented_control(
                "Who goes on the list",
                list(weekly.MEASURES),
                default=DEFAULT_MEASURE,
                key="weekly_measure",
                help="  \n".join(
                    f"**{name}:** {m.help}" for name, m in weekly.MEASURES.items()
                ),
            )
            or DEFAULT_MEASURE
        )
        m = weekly.MEASURES[measure]

        column = m.column
        h = weekly.headline(filtered_data, column)

        # Headline figures for the first week in the sidebar filters: usually next week, the
        # one there is still time to act on. The table below covers every week.
        first_week = h["first_week"]
        title = weekly.week_title(first_week)
        when_week = title[0].lower() + title[1:]
        week_end = (
            pd.to_datetime(filtered_data["Activity_Date"])
            .dt.to_period("W-SUN")
            .dt.end_time.dt.normalize()
        )
        next_week = filtered_data[week_end == first_week]
        hw = weekly.headline(next_week, column)
        plural = "s" if h["n_weeks"] != 1 else ""
        st.markdown(
            f"**{weekly.week_title(first_week)}** (week ending {weekly.week_label(first_week, year=True)})"
        )
        with st.container(horizontal=True):
            st.metric(
                f"Total appointments {when_week}",
                f"{hw['n']:,}",
                f"{hw['n'] - h['n'] / h['n_weeks']:+.0f} vs weekly average",
                delta_color="off",
                icon=":material/calendar_month:",
                border=True,
                help=f"In the week ending {weekly.week_label(first_week, year=True)}, in the sidebar "
                f"filters. The average over {h['n_weeks']} week{plural} is {h['n'] / h['n_weeks']:.0f}.",
            )
            vs_average = round(hw["total"] - h["weekly_average_total"])
            st.metric(
                m.count_label,
                f"{hw['total']:,.0f} ({hw['overall']:.0%})",
                f"{vs_average:+d} vs weekly average",
                delta_color="inverse",
                icon=":material/event_busy:",
                border=True,
                help=f"{m.help} The weekly average is {h['weekly_average_total']:.0f} "
                f"({h['overall']:.0%}).",
            )
            st.metric(
                "Highest-risk weekday",
                hw["top_day"],
                weekly.pts_vs_average(hw["top_day_rate"], hw["overall"])
                if hw["n_days"] > 1
                else None,
                delta_color="inverse",
                icon=":material/today:",
                border=True,
                help=f"{hw['top_day_rate']:.0%} on {hw['top_day']}, against {hw['overall']:.0%} "
                "for the week.",
            )
            expected_cost = costs.expected_dna_cost(filtered_data)
            week_cost = costs.expected_dna_cost(next_week)
            st.metric(
                f"Expected cost of missed appointments {when_week}",
                f"£{week_cost:,.0f}",
                f"{week_cost - expected_cost / h['n_weeks']:+,.0f} vs weekly average",
                delta_color="inverse",
                icon=":material/payments:",
                border=True,
                help=f"The average over {h['n_weeks']} week{plural} is £{expected_cost / h['n_weeks']:,.0f}. "
                "Each appointment's predicted risk of not attending times the cost of missing it, "
                "whichever count is chosen above. The costs are assumptions you can change under "
                "**Cost assumptions** in the sidebar.",
            )

        # The call list covers the first week: the period's counted appointments
        period = next_week
        when = when_week
        listed = period[period[column] == 1] if m.flag else period

        # Predicted savings from phoning everyone on the list
        savings_box(
            listed,
            MEASURE_RISK_MIN[column],
            f"Predicted savings {when} if we phone everyone on this list",
            f"If half of those who would have missed attend instead, with {len(listed):,} calls "
            f"in the week ending {weekly.week_label(first_week, year=True)}.",
            "not just this week or your filters",
            "week_savings",
            f"Nobody {when} is on this list, so there are no calls to cost.",
        )

        # The call list itself, highest risk first
        st.subheader("Calls to make")
        listed = listed.sort_values("DNA_Prob_Forecast", ascending=False)
        if m.flag:
            st.markdown(
                f"**{len(listed)} of {len(period)} appointments {when} are {m.day_phrase}**, highest "
                "risk first. Phone from the top."
            )
        else:
            st.markdown(
                f"**All {len(period)} appointments {when}**, highest predicted risk first. "
                "Phone from the top."
            )
        if period.empty:
            st.info("There are no appointments in the sidebar filters in this period.")
        elif listed.empty:
            st.info(f"None of the appointments {when} are {m.day_phrase}.")
        else:
            st.dataframe(
                appointment_table(listed, call_list=True),
                column_config=APPOINTMENT_COLUMNS,
                hide_index=True,
                key=f"call_list_{filter_key}",
            )
        st.caption(
            "To decide how many calls are worth making, see the **How many people should we phone?** tab. To see why "
            "the model gave an appointment its risk, choose **Show all appointments** in the sidebar."
        )


# ---------------------------------------------------------------- tab 3
# Ported from the "Who would we phone?" page of lambda_ml_streamlit_exercise. The counts
# come from past appointments with known outcomes; the upcoming appointments (all of them:
# the sidebar hides the filters on this tab) only turn the chosen setting into calls per week.
with tab_calls:
    meta = load_model_meta()
    past = load_past()
    p_past = past["DNA_Prob_Forecast"].to_numpy()

    # The thought-experiment box sits under the opening text and the notice column is wider than
    # on the other tabs, so the short opening text and the long notice come out about as tall
    with intro_columns("intro_calls", widths=(0.55, 0.45)):
        st.subheader("How many people should we phone?")
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

    # All the upcoming appointments, by week: the filters are hidden on this tab
    upcoming = data.assign(
        DNA_Cost=costs.appointment_costs(data, dna_costs, share_lost)
    )
    upcoming_week = pd.to_datetime(upcoming["Activity_Date"]).dt.to_period("W-SUN")
    per_week = upcoming_week.value_counts()
    n_weeks = len(per_week)

    # Starting values set through the session state rather than default=/value=, so the
    # savings button on the appointments tab can preset them without a Streamlit warning
    st.session_state.setdefault("call_mode", "Calls the team can make")
    st.session_state.setdefault("risk_min", round(HIGH_RISK_MIN * 100))
    st.session_state.setdefault("call_effect", DEFAULT_CALL_EFFECT)
    mode = (
        st.segmented_control(
            "Choose who to phone by",
            ["Calls the team can make", "Predicted risk"],
            key="call_mode",
        )
        or "Calls the team can make"
    )

    # The setting for who gets a call, and the guess at what a call changes, side by side.
    # The reflow_docked_ keys here and below let style.css stack rows where the sidebar
    # leaves the page narrow.
    ctrl_left, ctrl_right = st.container(key="reflow_docked_controls").columns(
        2, gap="large"
    )

    if mode == "Calls the team can make":
        # Slider runs up to the busiest upcoming week, rounded up to a multiple of 10. Clamp
        # the kept value first, so it is never out of range.
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
            "(from all the upcoming appointments).",
        )
        # Each week, phone that week's highest-risk appointments, up to capacity. The past
        # appointments then get the same share of calls, highest risk first.
        rank_in_week = (
            upcoming["DNA_Prob_Forecast"]
            .groupby(upcoming_week)
            .rank(method="first", ascending=False)
        )
        upcoming_called = rank_in_week <= cap
        called = calls.called_by_share(p_past, upcoming_called.mean())
        how = (
            "**This is the cut-off.** It isn't chosen directly: it follows from how many calls the team can "
            "make. More calls means a lower bar, so more patients get a call, and more of them are people "
            "who would have come anyway. **Move the slider and watch it change.**"
        )
    else:
        # Risk only: confidence is a function of the risk alone, so it adds nothing a risk bar doesn't, and
        # on its own it would also pick the patients the model is confident will attend.
        # Few risks pass 60% at an 8% DNA rate, so the slider stops there, in 1% steps
        risk_min = (
            ctrl_left.slider(
                "Phone if the predicted risk is at least",
                0,
                60,
                step=1,
                format="%d%%",
                key="risk_min",
            )
            / 100
        )
        called = calls.called_by_threshold(p_past, risk_min)
        upcoming_called = pd.Series(
            calls.called_by_threshold(upcoming["DNA_Prob_Forecast"], risk_min),
            index=upcoming.index,
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
        step=5,
        format="%d%%",
        key="call_effect",
        help="An assumption, not a measurement. Try a few values: 0% means the calls make no difference.",
    )
    ctrl_right.caption("A guess: nobody knows yet how many a call would change.")

    out = calls.call_outcomes(called, past["Did_Not_Attend"], p_past)

    # Weekly figures for all the upcoming appointments. Precision is None
    # when the setting phones nobody, and dna_cost is None when nobody upcoming is phoned.
    weekly_calls = upcoming_called.sum() / n_weeks
    dna_cost = costs.cost_per_dna_reached(upcoming[upcoming_called])
    has_costs = out["calls"] > 0 and weekly_calls > 0 and dna_cost is not None
    if out["calls"] > 0:
        weekly_reached = weekly_calls * out["precision"]
        weekly_avoided = weekly_reached * effect / 100
    if has_costs:
        econ = costs.call_economics(
            weekly_calls,
            weekly_reached,
            effect / 100,
            cost_per_call,
            minutes_per_call,
            dna_cost,
        )
        # None when the calls reach nobody who would miss: no effect could pay for them
        be = econ["break_even_effect"]
        be = float("inf") if be is None else be

    with st.container(border=True):
        k1, k2 = st.container(key="reflow_docked_cutoff").columns(
            [0.35, 0.65], gap="large"
        )
        k1.metric("Calls per 1,000 appointments", f"{out['calls']}")
        k2.metric(
            "So we phone anyone with a predicted risk of at least"
            if mode == "Calls the team can make"
            else "Lowest predicted risk that gets a call",
            f"{out['cutoff']:.0%}" if out["cutoff"] is not None else "nobody is phoned",
        )
        st.markdown(how)

    # Cost of the chosen setting; the assumptions and caveats sit in the cards' help and the caption
    if has_costs:
        with st.container(border=True):
            st.markdown(
                f"**What this setting costs and might save, per week** (about {weekly_calls:.0f} calls, "
                "for all the upcoming appointments)"
            )
            # Fixed columns rather than a horizontal container, so the four cards never wrap
            # three and one. Keyed so style.css shows them two per line on narrower screens.
            c1, c2, c3, c4 = st.container(key="reflow_halves_costs").columns(
                4, gap="small"
            )
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
                # A minus only for a loss, so st.metric still colours it red; no plus or arrow
                f"{'-' if econ['net'] < 0 else ''}£{abs(econ['net']):,.0f} after the cost of calls",
                delta_arrow="off",
                border=True,
                height="stretch",
                icon=":material/savings:",
                help=f"If a call gets {effect}% of the people it reaches, who would have missed, "
                f"to attend. A missed appointment among the patients these calls reach is valued at "
                f"about £{dna_cost:,.0f} on average. Change this and the call cost under **Cost "
                "assumptions** in the sidebar.",
            )
            c4.metric(
                "Missed appointments avoided",
                f"{weekly_avoided:.1f}"
                if weekly_avoided < 10
                else f"{weekly_avoided:.0f}",
                border=True,
                height="stretch",
                icon=":material/event_available:",
                help=f"About {weekly_reached:.0f} calls a week would reach someone who would otherwise "
                f"miss their appointment. If {effect}% of them attend instead, that's about "
                f"{weekly_avoided:.1f} fewer missed appointments a week: the assumption behind "
                "'Possibly saved'."
                + (
                    f" That's about {weekly_calls / weekly_avoided:.0f} calls, or "
                    f"£{weekly_calls / weekly_avoided * cost_per_call:,.0f} of staff time, for each "
                    "one avoided."
                    if weekly_avoided > 0
                    else ""
                ),
            )
            st.markdown(
                f"Your guess of {effect}% is {'above' if effect / 100 > be else 'below'} the "
                f"break-even, so the calls would {'more than ' if effect / 100 > be else 'not '}"
                "cover their cost."
                if be <= 1
                else "Even if every call got the patient to attend, the calls would cost more than "
                "the missed appointments they reach."
            )
            st.caption(
                "Estimates from the sidebar's cost assumptions. A missed appointment's cost is the value "
                "of a clinic slot, not cash that comes back. The call cost is staff time only."
            )
    elif out["calls"] > 0:
        st.info(
            "None of the upcoming appointments would get a call with this setting, so there is "
            "nothing to cost. Lower the bar."
        )

    if out["calls"] == 0:
        st.info(
            "With this setting nobody would get a call. Raise the number of calls or lower the bar."
        )
    else:
        # Keyed so style.css puts the cards under the chart on narrower screens
        left, right = st.container(key="reflow_stack_icons").columns(
            [0.62, 0.38], gap="large"
        )
        with left:
            # Share the 60 markers of width between the columns in line with their sizes; 60 makes
            # the chart wide and short enough to fill its column. The cap keeps the right one at
            # least as wide as its box.
            left_cols = min(44, max(8, round(60 * out["calls"] / 1000)))
            st.plotly_chart(
                charts.outcome_blocks(
                    out,
                    left_cols,
                    60 - left_cols,
                    heading="Using the model to rank the list",
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption(
                'Each marker is one of 1,000 past appointments. "On the call list" is the model\'s list (no '
                "calls were made); the boxes show what really happened. An unnecessary call went to "
                "someone who would have attended anyway."
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
            st.markdown(
                f"**If we phoned these {out['calls']} appointments in every 1,000:**"
            )
            # Short titles on the cards, with the explanation in a caption under each number
            with st.container(border=True, gap="small"):
                st.metric(
                    "Calls that reach a miss",
                    f"{out['precision']:.0%}",
                    f"{(out['precision'] - out['random_precision']) * 100:+.0f} pts vs random "
                    f"({out['random_precision']:.0%})",
                    icon=":material/call:",
                    help="Also called precision.",
                )
                st.caption(
                    f"{out['called_dna']} of the {out['calls']} calls would reach someone who then did not attend."
                )
            with st.container(border=True, gap="small"):
                st.metric(
                    "Misses reached",
                    f"{out['recall']:.0%}",
                    icon=":material/person_search:",
                    help="Also called recall or sensitivity.",
                )
                st.caption(
                    f"We would reach {out['called_dna']} of the {out['dna_total']} patients who did not attend. "
                    f"The other {out['not_called_dna']} would get no call."
                )
            with st.container(border=True, gap="small"):
                st.metric(
                    "Unnecessary calls",
                    f"{out['called_attended']}",
                    icon=":material/event_available:",
                )
                st.caption(
                    f"These people would have attended anyway: "
                    f"{out['called_attended'] / out['calls']:.0%} of the {out['calls']} calls."
                )
            st.caption(
                f"Counts are scaled from {meta['n_past_test']:,} past appointments to 1,000, so they can be out "
                "by a few. They count appointments that were missed without a call, not missed appointments a "
                "call would have prevented."
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

# ---------------------------------------------------------------- tab 0: Your rule vs the model
# First on the bar but last in the script: the training animation pauses between frames, so
# the other tabs are drawn before it starts. The visitor builds a rule from the clues the model
# sees, each with an equal say; then a machine learning model learns its own weighting from more and more
# past appointments. Everything is scored on the same past appointments as tab 3, scaled to
# 1,000, at the same number of calls (rule_game.CALL_SHARE). Built for a 2-5 minute visit at a
# stand: the clues are on screen without scrolling, and "Start again" clears it for the next person.
with tab_game:
    game_past, learning = load_rule_game()
    all_catches = load_rule_catches()
    n_rules = len(all_catches)
    best_possible = max(all_catches.values())
    frames = learning["frames"]
    n_missed = round(game_past["Did_Not_Attend"].mean() * 1000)
    n_calls = round(rule_game.CALL_SHARE * 1000)
    playing = any(k in st.session_state for k in ("game_best", "game_trained")) or bool(
        st.session_state.get("game_clues")
    )

    # ---- The setup: the title, the three numbers that matter, and the question
    with st.container(horizontal=True, vertical_alignment="center", key="game_title"):
        st.subheader(
            "Can you beat a computer at predicting who won't attend their appointment?"
        )
        if playing:
            st.button(
                "Start again",
                key="game_reset_top",
                on_click=reset_game,
                icon=":material/restart_alt:",
            )
    st.markdown(
        f"""<div class="game-stats">
        <div class="game-stat"><span class="game-stat-num">1,000</span>past appointments, when no reminder calls were made</div>
        <div class="game-stat"><span class="game-stat-num">{n_missed}</span>of them were missed</div>
        <div class="game-stat"><span class="game-stat-num">{n_calls}</span>calls a clinic team could have made</div>
        </div>
        <p class="game-question">Make a rule for who goes on the call list. <span class="highlight">How many of
        the {n_missed} who missed can you get on it?</span> Then a machine learning model will try.</p>""",
        unsafe_allow_html=True,
    )
    st.caption(
        ":material/science: Demonstration version: all the appointments here are synthetic. "
        "No real patients or Trust data are displayed."
    )

    # ---- 1. The visitor's rule
    st.markdown("#### 1. Make your rule")
    # A tinted box (style.css) rather than st.info, which can't carry the help tooltip
    with st.container(key="game_instruction"):
        st.markdown(
            ":material/touch_app: **What makes someone likely to miss? Pick one clue or several.** "
            "Each clue you pick gets an equal say; the model will get to choose its own.",
            # Worked example from the real scores in data/rule_game_past.csv (rule_game.rule_score)
            help=(
                "**How an equal say works.** Each clue gives every patient a score from 0 to 1, by where "
                "they rank on that clue: 1 is the top of the ranking. Your rule adds up the scores for the "
                "clues you picked, and the highest totals are phoned first.\n\n"
                "**Example:** you pick *most missed appointments before* and *being younger*.\n\n"
                "- **Patient A** has missed 3 appointments before; only 5% of patients have missed more, "
                "so A scores 0.95. A is 70, younger than only about 16% of patients: 0.17. "
                "**Total 1.12.**\n"
                "- **Patient B** has never missed one. Two-thirds of patients haven't, and they share the "
                "bottom of that ranking: 0.33. B is 25, younger than about 93% of patients: 0.94. "
                "**Total 1.27.**\n\n"
                "So B goes above A on the list, even though A has missed 3 appointments: with an equal "
                "say, being young counts as much as having missed before."
            ),
        )
    clues = st.pills(
        "Put these patients first on the call list:",
        options=list(rule_game.CLUES),
        format_func=lambda clue: rule_game.CLUES[clue],
        selection_mode="multi",
        key="game_clues",
    )

    if not clues:
        st.info(
            "Pick at least one clue to see who your rule would put on the call list.",
            icon=":material/touch_app:",
        )
    else:
        mine = rule_game.results(game_past, rule_game.rule_score(game_past, clues))
        caught = mine["outcomes"]["called_dna"]
        # The change since the visitor's previous rule, so adding a clue shows what it did
        rule_key = sorted(clues)
        last = st.session_state.get("game_last")
        if last is None or last["clues"] != rule_key:
            st.session_state["game_prev"] = last
            st.session_state["game_last"] = {"clues": rule_key, "caught": caught}
        prev = st.session_state.get("game_prev")
        best = st.session_state.get("game_best")
        if best is None or mine["reached"] > best["reached"]:
            st.session_state["game_best"] = {"clues": list(clues), **mine}

        left, right = st.container(key="reflow_stack_game_rule").columns(
            [0.62, 0.38], gap="large"
        )
        with left:
            st.plotly_chart(
                # 50 wide on the right so the chart is wide and short enough to fill its column
                charts.outcome_blocks(
                    mine["outcomes"], 10, 50, heading="Using your rule to rank the list"
                ),
                width="stretch",
                config={"displayModeBar": False},
                key="game_rule_icons",
            )
            st.caption(
                "Each marker is one of the 1,000 past appointments. Nobody was phoned: your rule would "
                f"have put the left-hand ones in the first {n_calls}. Missed or attended is what really happened."
            )
        with right:
            st.metric(
                "Missed appointments correctly predicted by your rule",
                f"{caught} of {n_missed}",
                f"{caught - prev['caught']:+d} since your last rule" if prev else None,
                border=True,
                icon=":material/call:",
            )
            # What the last change did: the hook is seeing a score fall when a clue is added
            if prev:
                added = [c for c in rule_key if c not in prev["clues"]]
                removed = [c for c in prev["clues"] if c not in rule_key]
                change = (
                    "Adding "
                    + " and ".join(
                        f"*{rule_game.CLUE_SHORT[c].lower()}*" for c in added
                    )
                    if added and not removed
                    else "Taking away "
                    + " and ".join(
                        f"*{rule_game.CLUE_SHORT[c].lower()}*" for c in removed
                    )
                    if removed and not added
                    else "Changing your rule"
                )
                gain = caught - prev["caught"]
                if gain < 0:
                    st.error(
                        f"**{change} made your list worse**, from {prev['caught']} to {caught}."
                        + (
                            " More information, a worse list."
                            if added and not removed
                            else ""
                        ),
                        icon=":material/trending_down:",
                    )
                elif gain > 0:
                    st.success(
                        f"**{change} made your list better**, from {prev['caught']} to {caught}.",
                        icon=":material/trending_up:",
                    )
                else:
                    st.info(
                        f"**{change} made no difference.** Still {caught}.",
                        icon=":material/trending_flat:",
                    )
            st.metric(
                "Your rule ranks",
                f"#{rule_game.rank_of(caught, all_catches)} of {n_rules}",
                border=True,
                icon=":material/leaderboard:",
                help=f"Out of all {n_rules} rules you can make with these buttons, scored the same way.",
            )
            best = st.session_state["game_best"]
            st.markdown(
                f"**Your best rule so far** would have reached {best['outcomes']['called_dna']} of {n_missed}: "
                + ", ".join(f"*{rule_game.CLUES[c].lower()}*" for c in best["clues"])
                + "."
            )
            st.caption(
                "Try adding or taking away a clue. Does more information always help?"
            )

    # ---- 2. The model learns
    st.divider()
    st.markdown("#### 2. Watch a machine learning model learn its own rule")
    st.markdown(
        f"It gets the same {len(rule_game.CLUES)} clues, and **{learning['n_train']:,} other past "
        "appointments** with who missed them. It adds up the clues just as your rule does, but nobody tells it how much say each "
        "clue gets or which way it points: it works that out. Then "
        "it makes its own list for the same 1,000 appointments, which it has never seen."
    )
    best = st.session_state.get("game_best")
    run = st.button(
        "Start learning",
        icon=":material/play_arrow:",
        type="primary",
        disabled=best is None,
        help=None if best else "Make a rule first, so there is something to beat.",
        key="game_run",
    )
    # Bars in the order of the model's final say, so they don't jump between frames
    say_order = sorted(rule_game.CLUES, key=lambda f: -frames[-1]["say"][f])
    at = rule_game.overtook_at(frames, best["reached"]) if best else None
    slot = st.empty()

    def draw_model(i, phase, best):
        """One frame: the scores, then the race (left) beside how much say each clue gets (right)."""
        f = frames[i]
        with slot.container():
            m1, m2, m3 = st.container(key=f"reflow_halves_game_{phase}_{i}").columns(3)
            m1.metric(
                "Past appointments learnt from",
                f"{f['n']:,}",
                border=True,
                height="stretch",
                icon=":material/school:",
            )
            m2.metric(
                "Missed appointments correctly predicted by the computer",
                f"{f['outcomes']['called_dna']} of {n_missed}",
                border=True,
                height="stretch",
                icon=":material/smart_toy:",
            )
            m3.metric(
                "Missed appointments correctly predicted by your best rule",
                f"{best['outcomes']['called_dna']} of {n_missed}",
                border=True,
                height="stretch",
                icon=":material/person:",
            )
            left, right = st.container(key=f"reflow_stack_game_{phase}_{i}").columns(
                [0.55, 0.45], gap="large"
            )
            with left:
                st.markdown("**The race: missed appointments correctly predicted**")
                st.plotly_chart(
                    charts.learning_curve(
                        frames,
                        i,
                        best["outcomes"]["called_dna"],
                        learning["perfect"]["outcomes"]["called_dna"],
                        n_missed,
                        overtook=at,
                        height=440,
                    ),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"game_curve_{phase}_{i}",
                )
            with right:
                st.markdown("**How much say each clue gets**")
                st.plotly_chart(
                    charts.say_bars(
                        say_order,
                        rule_game.CLUE_SHORT,
                        f["say"],
                        clues or best["clues"],
                    ),
                    width="stretch",
                    config={"displayModeBar": False},
                    key=f"game_say_{phase}_{i}",
                )

    if run and best:
        for i in range(len(frames)):
            draw_model(i, "run", best)
            if i == at:
                st.toast(
                    "The model just overtook your best rule!",
                    icon=":material/emoji_events:",
                )
            if i < len(frames) - 1:
                # Linger where the model passes you, hurry through frames where little changes
                flat = (
                    abs(frames[i + 1]["reached"] - frames[i]["reached"]) * n_missed
                    <= 1.5
                )
                time.sleep(2.0 if i == at else 0.4 if flat else 0.9)
        st.session_state["game_trained"] = True
    elif st.session_state.get("game_trained") and best:
        draw_model(len(frames) - 1, "done", best)

    # ---- 3. The verdict, straight after the race
    if st.session_state.get("game_trained") and best:
        st.markdown("#### 3. See whose list did best")
        final, perfect = frames[-1], learning["perfect"]
        yours, theirs = best["outcomes"]["called_dna"], final["outcomes"]["called_dna"]
        st.success(
            (
                f"**The model overtook your best rule after learning from {frames[at]['n']:,} past "
                "appointments.** "
                if at is not None
                else ""
            )
            + f"With all {learning['n_train']:,}, the computer would have correctly predicted {theirs} of "
            f"the {n_missed} missed appointments, against your {yours}.",
            icon=":material/emoji_events:",
        )

        say = final["say"]
        top_two = say_order[:2]
        found = (
            "\n\nYou found the best rule the buttons can make, and it still won."
            if yours >= best_possible
            else f"\n\nNo rule from the buttons can match it: the best reaches {best_possible} of {n_missed}."
        )

        c1, c2 = st.container(key="reflow_stack_game_verdict_cards").columns(
            2, gap="small"
        )
        with c1.container(border=True, key="title_card_game_say", height="stretch"):
            st.markdown(
                ":material/balance: **Same clues, but the model can learn which matter most as it goes.**"
            )
            st.markdown(
                "Your rule gives every clue an equal say - tuning it more by hand is too hard. "
                ""
                "\n\nThe model gives "
                f"*{rule_game.CLUE_SHORT[top_two[0]].lower()}* {say[top_two[0]]:.0%} and "
                f"*{rule_game.CLUE_SHORT[top_two[1]].lower()}* {say[top_two[1]]:.0%}, and the rest less."
                # " {found}"
            )
        with c2.container(border=True, key="title_card_game_limit", height="stretch"):
            st.markdown(":material/visibility_off: **Nobody gets them all.**")
            st.markdown(
                "Even a perfect list, one that knows each patient's true chance of missing, reaches only "
                f"{perfect['outcomes']['called_dna']} of {n_missed}."
                "\n\nMany misses come down to things no data records, like bad weather or bus delays."
            )

        # ---- Why a computer? (like the geographic mini's "Why computers?")
        st.markdown("#### Why use a computer instead of a human here?")
        n_clues = len(rule_game.CLUES)
        w1, w2 = st.container(key="reflow_halves_game_why").columns(2, gap="small")
        w1.metric(
            "Computers can learn from **huge** amounts of past data - here it learnt from",
            f"{learning['n_train']:,} past appointments",
            border=True,
            height="stretch",
        )
        w2.markdown(
            "- Computers can re-learn when things change, like a new site or new booking habits. "
            "\n- Some models can also spot clues that matter more in combination, like a long gap mattering more "
            "before a first appointment, or Monday mornings being worse than other days.",
        )

        # st.markdown(
        #     "- **It learns how much each clue counts, and which way it points,** from what really happened, "
        #     f"not from a hunch, and weighs all {n_clues} the same way for every patient.\n"
        #     "Remeber - it can only go by what's in the data.** It can't see a bus strike, or bad weather, "
        #     "or whether patients are parents who can't make 9am appointments, and it "
        #     "learns past patterns, unfair ones included. "
        #     ""
        #     "Predictions support staff judgement rather than replace it."
        # )
        # st.button(
        #     "Start again for the next person",
        #     key="game_reset_end",
        #     on_click=reset_game,
        #     icon=":material/restart_alt:",
        # )

    # ---- The way on: styled like the geographic mini's next-step boxes (style.css)
    st.divider()
    with st.container(key="game_next"):
        onward, plan = st.container(key="reflow_stack_game_next").columns(
            2, gap="large"
        )
        onward.markdown(
            "### See the model at work\n"
            "The tool this demo is based on gives every upcoming appointment a predicted risk. See which "
            "ones it flags for staff to call, and why."
        )
        onward.button(
            "See upcoming appointments",
            key="to_appts",
            on_click=go_to_tab,
            args=(TAB_LABELS[1],),
            icon=":material/arrow_forward:",
            type="primary",
        )
        plan.markdown(
            "### Or plan the calls\n"
            "How many calls should the team make, and would they be worth the staff time?"
        )
        plan.button(
            "Plan the calls",
            key="to_calls",
            on_click=go_to_tab,
            args=(TAB_LABELS[2],),
            icon=":material/arrow_forward:",
        )

# After a next-tab button, open the new tab at the top (from the geographic mini)
if st.session_state.pop("scroll_to_top", False):
    st.html(
        "<script>setTimeout(function () {"
        "var m = document.querySelector('[data-testid=\"stMain\"]'); if (m) m.scrollTo(0, 0);"
        "window.scrollTo(0, 0);}, 100);</script>",
        unsafe_allow_javascript=True,
    )
