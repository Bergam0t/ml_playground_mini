import base64
import hashlib
import io
import json
import random
import time
from typing import Any
from pathlib import Path

import streamlit as st
import pandas as pd

import calls
import charts
import click_hint
import costs
import explain
import grow_waterfall
import move_markers
import pilot
import rule_game
import weekly
from demo_data import (
    DATA_DIR,
    TRUST_NAME,
    load_forecast,
)  # DEMO: synthetic data replaces SQL
from mobile_nav import add_mobile_nav

HSMA_URL = "https://hsma.co.uk"
REAL_TOOL_CODE_URL = "https://github.com/WX-BIU/Outpatient-DNAs"
CODE_URL = "https://github.com/hsma-programme/ml_playground_mini"

st.set_page_config(
    page_title="Outpatient DNA Forecast",
    layout="wide",
    initial_sidebar_state="expanded",
)


# The widest Streamlit ever sends an image to the browser (its MAXIMUM_CONTENT_WIDTH)
MAX_IMAGE_WIDTH = 1460


@st.cache_resource
def fitted_image(path):
    """The PNG at `path` as Streamlit would send it: shrunk to MAX_IMAGE_WIDTH if wider, in the
    same way. Streamlit would do that on every run, which for the two big logos took about 0.4s
    of each page load; this does it once."""
    from PIL import Image

    image = Image.open(path)
    if image.width > MAX_IMAGE_WIDTH:
        height = int(1.0 * image.height * MAX_IMAGE_WIDTH / image.width)
        image = image.resize((MAX_IMAGE_WIDTH, height), resample=Image.BILINEAR)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


st.logo(
    fitted_image(
        "resources/Chalky HSMA-07-wide white logo tall with tagline - new tagline.png"
    ),
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


# Past appointments the model didn't train on, with what really happened (for the phone tab)
@st.cache_data
def load_past():
    return pd.read_csv(DATA_DIR / "past_predictions.csv")


def catch_rates(threshold):
    """On the past appointments the model didn't train on: the share of the missed appointments
    at or above this risk, and the share of those at or above it that were missed."""
    past = load_past()
    picked = past["DNA_Prob_Forecast"] >= threshold
    missed = past["Did_Not_Attend"] == 1
    return (picked & missed).sum() / missed.sum(), missed[picked].mean()


@st.cache_data
def load_model_meta():
    return json.loads((DATA_DIR / "model_meta.json").read_text(encoding="utf-8"))


# The bonus game tab: past appointments with a score per clue, and the
# model's results as it learns (both precomputed by build/rule_game_data.py)
@st.cache_data
def load_rule_game():
    return rule_game.load_past(), rule_game.load_learning()


# The same calls three ways (at random, by the simple rule, by the model), for the comparison box
# on the first tab; and the simple rule's score for each past appointment, for the savings figures.
# The rule game's past appointments are the same rows as past_predictions.csv, in the same order
# (build/rule_game_data.py checks this).
@st.cache_data
def load_comparison():
    game_past = rule_game.load_past()
    risk = load_past()["DNA_Prob_Forecast"].to_numpy()
    assert len(game_past) == len(risk), "rule game and past predictions don't line up"
    rule = rule_game.rule_score(game_past, rule_game.SIMPLE_RULE)
    return rule_game.three_ways(game_past, risk), game_past, rule


# The full outcomes behind each of the three ways' tiles, for the chart under them
@st.cache_data
def load_way_outcomes():
    game_past = rule_game.load_past()
    risk = load_past()["DNA_Prob_Forecast"].to_numpy()
    return rule_game.three_ways_outcomes(game_past, risk)


# What every rule the visitor can make would catch, for "Your rule ranks #N of 4,095"
@st.cache_data
def load_rule_catches():
    return rule_game.all_rule_catches(rule_game.load_past())


# Fewer filtered appointments than this and their factor chart is too noisy to show
MIN_GROUP_FOR_CHART = 30

# Rows shown at once in the All appointments table (it scrolls), and their height in px
TABLE_ROWS = 10
TABLE_ROW_HEIGHT = 35

# The "How to read this chart" steps; the numbers match the badges on charts.guide_waterfall()
GUIDE_STEPS = [
    (
        "<b>Start at the top: the average appointment.</b> This is the model's predicted risk "
        "of not attending, averaged over all appointments."
    ),
    (
        "<b>Each bar is one fact about this appointment.</b> "
        f"<b style='color:{charts.UP}'>Red</b> pushes the risk of not attending up, "
        f"<b style='color:{charts.DOWN}'>blue</b> pushes it down. The number is how far it "
        "moves, in percentage points. The biggest come first."
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
# The Very High band and the high-risk flag: the live app's 70%. The model is calibrated, so
# these appointments are missed about 84% of the time, like the real pilot's high-risk patients
HIGH_RISK_MIN = 0.70
# The share of the people a call reaches, who would have missed, that it gets to attend (%):
# the starting value on the phone tab and in the savings on the first tab and the demo dashboard. The real
# pilot's, about 80% (pilot.EFFECT), to the nearest 5 so it sits on the phone tab's slider.
DEFAULT_CALL_EFFECT = round(pilot.EFFECT * 20) * 5
# Its slider, the same on the first tab and the phone tab (both under the key call_effect)
CALL_EFFECT_SLIDER: dict[str, Any] = {
    "label": "Suppose a call gets this share of the people it reaches, who would have missed, "
    "to attend instead",
    "min_value": 0,
    "max_value": 100,
    "step": 5,
    "format": "%d%%",
    "help": f"Starts at the {pilot.SERVICE.lower()} pilot's figure: of about "
    f"{pilot.CALLS * pilot.DNA_BEFORE:.0f} patients who would have missed, {pilot.EXTRA_ATTENDED} "
    f"attended ({pilot.EFFECT:.0%}). That was one service, for patients at very high risk, with no "
    "comparison group, so other settings may see less. 0% means the calls make no difference.",
}


def in_words(share):
    """A share as "N in 10", or "1 in N" when it's small (0.15 is "1 in 7", not "1 in 10")."""
    if share < 0.3:
        return f"1 in {round(1 / share)}"
    return f"{round(share * 10)} in 10"


def bars_caveat():
    """What an explanation chart's bars mean, for a non-technical reader."""
    return (
        "\n\nThese bars show :yellow[**patterns**] the model spotted, :yellow[**not causes**]: "
        "changing one of these facts wouldn't necessarily make a patient more likely to attend. "
        "For that, we need a different kind of machine learning."
    )


def model_accuracy(question="How good is the model?"):
    """How good the model is, for a non-technical reader: how often a Very High flag was right,
    on past appointments the model hadn't seen, and why the records can't explain every missed
    appointment."""
    _, precision = catch_rates(HIGH_RISK_MIN)
    return (
        f"{question} We checked it on past appointments it hadn't seen. Of the patients it "
        f"rated Very High risk ({HIGH_RISK_MIN:.0%} or more), about {in_words(precision)} did "
        "miss.\n\nBut it only knows what's in the records. :yellow[**Many appointments are missed "
        "for reasons the records don't show**], such as a sudden illness or a problem with transport, "
        "so plenty of patients who miss never get a high risk."
    )


def model_caveat():
    """The explanation overlay's note on what the bars mean and how good the model is."""
    return f"{bars_caveat()} {model_accuracy()}"


def DNA_Risk(prob):
    if prob < 0.05:
        return "Low"
    elif prob < FORECAST_MIN:
        return "Medium"
    elif prob < HIGH_RISK_MIN:
        return "High"
    else:
        return "Very High"


# Where the confidence bands change (the popover's chart draws the same bands)
CONFIDENCE_EDGES = (0.5, 0.8, 0.9)


def Confidence_Score_Txt(confidence):
    medium, high, very_high = CONFIDENCE_EDGES
    if confidence < medium:
        return "Low"
    elif confidence < high:
        return "Medium"
    elif confidence < very_high:
        return "High"
    else:
        return "Very High"


# The table's forecast filters (above the appointments table): the bands from DNA_Risk()
# and Confidence_Score_Txt()
BANDS = ["Low", "Medium", "High", "Very High"]
RISK_HELP = (
    "The model's predicted risk that the patient won't attend.  \n"
    "**Low:** under 5%  \n**Medium:** 5% to 20%  \n**High:** 20% to 70%  \n"
    "**Very High:** 70% or more  \n"
    "About 8% of appointments are missed. High and Very High (20% or more, about two and a half "
    "times the average) are forecast as DNAs; Very High is also flagged high-risk."
)
# Only the bands: what confidence means, with examples, is in the "What does confidence
# mean?" popover beside it, so the two don't say the same thing
CONFIDENCE_HELP = (
    "Confidence bands, and the predicted risk each covers.  \n"
    "**Low:** under 50% (risk 8% to 43%)  \n"
    "**Medium:** 50% to 80% (risk 3% to 8% or 43% to 69%)  \n"
    "**High:** 80% to 90% (risk 1.3% to 3% or 69% to 83%)  \n"
    "**Very High:** 90% or more (risk under 1.3% or over 83%)"
)


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
    # The call list leaves out Expected DNA and Confidence
    if "Expected DNA" in df:
        styles["Expected DNA"] = df["Expected DNA"].map(EXPECTED_DNA_COLOURS).map(css)
    if "Confidence" in df:
        styles["Confidence"] = df["Confidence"].map(CONFIDENCE_COLOURS).map(css)
    # Same bands as the table's "DNA Risk" filter
    styles["Probabilty"] = (
        (df["Probabilty"] / 100).map(DNA_Risk).map(RISK_COLOURS).map(css)
    )
    return styles


# Appointment tables (the all-appointments view, a day's list in the weekly summary, and the call list):
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

# Fixed column widths (about 1,110px in total; about 1,185px on the call list, which adds three
# columns and drops two) so the whole table fits a 1,400px main area; the grid stretches them
# to fill any space left over.
APPOINTMENT_COLUMNS = {
    "Appointment Date": st.column_config.DateColumn(
        "Appointment Date", format="DD/MM/YYYY", width=110
    ),
    "Day": st.column_config.TextColumn(width=95),
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
        help="Model forecast: how clear-cut the yes-or-no call is. Low for risks near "
        f"{FORECAST_MIN:.0%}, where the forecast changes from No to Yes. "
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


# The call list's column settings: the same, with its date column headed "Date"
CALL_LIST_COLUMNS = {
    **{k: v for k, v in APPOINTMENT_COLUMNS.items() if k != "Appointment Date"},
    "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY", width=110),
}


# The call list only: the admin team needs a name and a time to phone and remind. Not shown in
# the all-appointments view
CALL_LIST_FIELDS = {
    "Appointment_Time": "Time",
    "Patient Name": "Patient Name",
}


def appointment_table(
    df, by_patient=False, call_list=False, highlight_row=None, flag=None
):
    """Appointments to show in a table, with the forecast columns coloured.

    `by_patient` puts the patient columns first and shades every other patient's rows,
    for a table sorted so each patient's appointments are together. `call_list` adds the
    day, the appointment time and the patient's name, and leaves out the Expected DNA and Confidence
    columns (the list is already everyone at or above the chosen risk). `highlight_row` is the position of a row to
    shade (outside the coloured forecast columns), as the table's cell selection only
    outlines the one cell clicked. `flag` is a True or False per row of `df` (in order): the
    True rows are tinted yellow (outside the coloured forecast columns).
    """
    fields = list(APPOINTMENT_FIELDS)
    if call_list:
        df = df.assign(Day=pd.to_datetime(df["Activity_Date"]).dt.day_name())
        fields.insert(1, "Day")
        fields.insert(2, "Appointment_Time")
        fields.insert(4, "Patient Name")
        fields = [f for f in fields if f not in ("DNA_Forecast_Text", "Confidence")]
    if by_patient:
        patient = ["MRN", "Mobile Telephone"]
        fields = patient + [f for f in fields if f not in patient]
    names = {**APPOINTMENT_FIELDS, **CALL_LIST_FIELDS}
    if call_list:
        names["Activity_Date"] = (
            "Date"  # the list is one week, so the short header will do
        )
    shown = df[fields].rename(columns=names)
    styler = shown.style.apply(forecast_cell_styles, axis=None)
    if by_patient:
        styler = styler.apply(patient_bands, axis=None)
    if call_list:
        styler = styler.apply(day_bands, axis=None)
    if flag is not None:
        styler = styler.apply(flagged_row_style, flag=list(flag), axis=None)
    if highlight_row is not None:
        styler = styler.apply(selected_row_style, position=highlight_row, axis=None)
    return styler


def flagged_row_style(df, flag):
    """Tint the rows marked True in `flag` yellow (outside the coloured forecast columns)."""
    plain = [
        c for c in df.columns if c not in ("Expected DNA", "Confidence", "Probabilty")
    ]
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    rows = [i for i, f in enumerate(flag) if f]
    styles.iloc[rows, [df.columns.get_loc(c) for c in plain]] = (
        "background-color: rgba(255, 230, 0, 0.22);"
    )
    return styles


def selected_row_style(df, position):
    """Shade the row at `position` (outside the coloured forecast columns)."""
    plain = [
        c for c in df.columns if c not in ("Expected DNA", "Confidence", "Probabilty")
    ]
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if 0 <= position < len(df):
        styles.iloc[position, [df.columns.get_loc(c) for c in plain]] = (
            "background-color: #3d4f5c;"
        )
    return styles


def day_bands(df):
    """Shade every other day's rows a faint dark grey (outside the coloured forecast columns),
    for a table sorted by date so each day's rows are together."""
    day_no = (df["Date"] != df["Date"].shift()).cumsum()
    plain = [
        c for c in df.columns if c not in ("Expected DNA", "Confidence", "Probabilty")
    ]
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles.loc[day_no % 2 == 0, plain] = "background-color: #2e2c2c;"
    return styles


def patient_bands(df):
    """Shade every other patient's rows (outside the coloured forecast columns)."""
    patient_no = (df["Patient ID"] != df["Patient ID"].shift()).cumsum()
    plain = [
        c for c in df.columns if c not in ("Expected DNA", "Confidence", "Probabilty")
    ]
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    styles.loc[patient_no % 2 == 0, plain] = "background-color: #2e2c2c;"
    return styles


POSTER_URL = "https://hsma.co.uk/impact_posters/Outpatient%20non%20attendance.png"
VIDEO_URL = "https://youtu.be/tWcLIMTQb8A"


def learn_more():
    """The HSMA "want to learn" note, ported from the geographic mini's footer, under the
    poster and the video."""
    st.divider()
    qr, text = st.columns([1, 5], vertical_alignment="center")
    qr.image("resources/hsma_qr.png", width=130)
    text.markdown(
        f"""
        **Want to learn how to build tools like this?**

        The Health Service Modelling Associates (HSMA) programme trains people working in
        health and social care in modelling, machine learning
        and data science. Scan the code or visit the
        [HSMA website]({HSMA_URL}) to find out more, look at
        [the code for the real tool]({REAL_TOOL_CODE_URL}), or
        [explore the code for this app]({CODE_URL}).
        """
    )


@st.dialog(
    "Read about the impact of the real tool this is based on, made by a member of the "
    "6th cohort of the Health Service Modelling Associates (HSMA) programme",
    width="large",
)
def show_poster():
    """The HSMA impact poster, in a pop-up so the visitor stays in the app."""
    st.image(POSTER_URL)
    st.caption(f"[Open full size in a new tab]({POSTER_URL})")
    learn_more()


@st.dialog(
    "Hear from the person who made the real tool this is based on, who was a member of the "
    "6th cohort of the Health Service Modelling Associates (HSMA) programme",
    width="large",
)
def show_video():
    """The video from the person who made the real tool, about the project, in a pop-up."""
    st.video(VIDEO_URL)
    learn_more()


def overall_importance(filtered_data, filter_key, sidebar_choices):
    """The "Overall" tab of the explanation overlay: what the model relies on across all upcoming
    appointments, and, when the sidebar filters narrow them enough, for the filtered ones beside
    it. `sidebar_choices` is (label, choice) for the filters that name a kind of appointment."""
    data = load_data()
    # filtered_data leaves out the table's forecast filters: they pick appointments by the
    # very pushes being charted (Very High risk = big pushes up), so the comparison would be
    # circular
    shap_all = load_shap()
    n = len(filtered_data)
    is_filtered = n < len(data)
    show_filtered = is_filtered and n >= MIN_GROUP_FOR_CHART

    st.markdown(
        "These bars show what the model relies on **on average, across many appointments**. "
        "Each bar is how much a factor moves the predicted risk of not attending, "
        "**whether it pushes the risk up or down**, so a longer bar means the model relies on "
        "that factor more.\n\nThis is different from the chart for one appointment, which shows "
        "how **that patient's own values moved** their risk, and in **which direction**. "
        "Hover over a bar for the figure."
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
        same_value = [label for label, choice in sidebar_choices if choice != "All"]
        if same_value:
            listed = " and ".join(same_value)
            notes.append(
                f"Every appointment in your filters has the same {listed}, so "
                f"{'that bar shows' if len(same_value) == 1 else 'those bars show'} how much "
                f"that one {listed} moves the risk."
            )
        st.markdown("\n".join(f"- {note}" for note in notes))
    elif is_filtered:
        st.caption(
            f"Only {n:,} appointment{'s' if n != 1 else ''} match your filters: too few for a "
            f"reliable chart of their own (it needs at least {MIN_GROUP_FOR_CHART})."
        )
    # Keyed "caveat" so style.css colours its highlight
    st.container(key=f"caveat_{filter_key}").info(model_caveat())
    st.warning(
        "This is a synthetic dataset, so these results should not be used for service planning.",
        icon=":material/science:",
    )


def open_explanation():
    """The appointments table's on_select: a row was picked (or unpicked), so the script opens
    the explanation if a row is now selected. Only on this run, so it doesn't reopen whenever
    the page reruns while the row stays selected."""
    st.session_state["explain_open"] = True


def close_explanation():
    """On dismissing the explanation, clear the table's selection (a new table key), so the
    same row can be clicked again."""
    st.session_state["table_nonce"] = st.session_state.get("table_nonce", 0) + 1


def explain_appointment(row, overall):
    """The SHAP waterfall for one appointment, in an overlay so it can't be missed below the
    table. Opened by clicking a row in the All appointments table. The overlay's title names
    the patient, so the dialog is made here, when it opens, rather than with a fixed title."""
    meta = load_model_meta()
    expl = explain.patient_explanation(
        row, load_shap().loc[row["row_id"]], meta["base_value_pp"]
    )
    title = (
        f"Patient {row['MRN']} on {row['Activity_Date']:%d/%m/%Y} has a "
        f"{expl['fx']:.0f}% risk of not attending: why?"
    )
    st.dialog(title, width="large", on_dismiss=close_explanation)(explanation_body)(
        row, expl, overall
    )


def explanation_body(row, expl, overall):
    """What the explain_appointment overlay shows: this appointment, and, on a second tab, what
    the model relies on overall."""
    # A segmented control, not st.tabs: style.css turns every top-level tab bar into the page
    # navbar, and a dialog's would be caught by that
    show = st.segmented_control(
        "Show",
        ["This appointment", "Overall: what the model relies on"],
        default="This appointment",
        required=True,
        key="explain_tab",
        label_visibility="collapsed",
    )
    if show == "This appointment":
        this_appointment(row, expl)
    else:
        overall_importance(*overall)


def how_to_read(key, stacked=False):
    """The "How to read this chart" guide: a made-up example waterfall and the numbered steps.
    Side by side in the explanation overlay; `stacked` puts the example above the steps, for
    the narrow column beside the chart on the first tab. `key` keeps the example's chart apart."""
    steps = "\n".join(
        f"<div class='guide-step'><span class='guide-num'>{n}</span><div>{text}</div></div>"
        for n, text in enumerate(GUIDE_STEPS, start=1)
    )
    if stacked:
        example_col = steps_col = st.container()
    else:
        # Keyed so style.css stacks the example above the steps on narrower screens
        example_col, steps_col = st.container(key=f"reflow_stack_{key}").columns(
            [4, 5], vertical_alignment="center"
        )
    example_col.plotly_chart(
        charts.guide_waterfall(),
        key=key,
        width="content",
        config={"staticPlot": True},
    )
    steps_col.markdown(steps, unsafe_allow_html=True)


def patient_waterfall(expl, key, token=None, grow=False):
    """One appointment's waterfall and a sentence naming its two biggest contributions. `grow`
    plays the chart from empty (grow_waterfall.py), for a newly picked patient; `token` tells
    that script this chart from the one before it."""
    st.plotly_chart(
        charts.waterfall(expl, meta=token),
        key=key,
        width="stretch",
        config={"displayModeBar": False},
    )
    if grow:
        grow_waterfall.grow_waterfall(expl, token, key)
    top = expl["shown"][:2]
    st.markdown(
        f"The biggest contributions are **{top[0]['label']}** ({top[0]['value']:+.1f} points) and "
        f"**{top[1]['label']}** ({top[1]['value']:+.1f} points)."
    )


def this_appointment(row, expl):
    """The "This appointment" tab: the waterfall for one appointment."""
    # Starts open; collapse it once you have the idea (it stays collapsed between patients)
    with st.expander(
        "How to read this chart (a made-up example)",
        expanded=True,
        key="chart_guide",
        icon=":material/help:",
    ):
        how_to_read("guide")

    patient_waterfall(expl, f"waterfall_{row['row_id']}")
    # Keyed "caveat" so style.css colours its highlight
    st.container(key="caveat_appointment").info(
        "A predicted risk is an estimate based on similar appointments in past data, not a "
        f"forecast of what this one person will do. {model_caveat()}"
    )
    st.warning(
        "This is not a real patient, and the factors shown here should not be used for "
        "service planning.",
        icon=":material/science:",
    )


@st.fragment
def appointments_table(table_data, filter_key, sidebar_choices):
    """The All appointments table, with its own forecast filters, highest risk first. A
    fragment, so clicking a row (which opens explain_appointment) or changing these two filters
    reruns just this part, not the whole page. The sidebar's filters are outside it: changing
    one reruns the page, which passes the newly filtered appointments in here."""
    filtered_data = table_data  # the sidebar's filters, before this table's own
    st.subheader("Appointment details")
    with st.container(key="click_cta_all"):
        st.info(
            "**Click a row** (highest risk first) to see why the model gave that appointment its "
            "risk of not attending.",
            icon=":material/touch_app:",
        )

    # The model's forecast filters, for this table only (the rest of the page follows
    # just the sidebar's filters), with the confidence explainer on the same row: a
    # popover floats over the page, so it doesn't push the table down as an expander would
    risk_col, confidence_col, about_col = st.columns(
        [2, 2, 1.3], vertical_alignment="bottom"
    )
    selected_dna_risk = risk_col.selectbox(
        "DNA Risk", ["All", *BANDS], key="filter_risk", help=RISK_HELP
    )
    selected_confidence = confidence_col.selectbox(
        "Confidence", ["All", *BANDS], key="filter_confidence", help=CONFIDENCE_HELP
    )
    with about_col.popover(
        "What does confidence mean?", icon=":material/help:", width="stretch"
    ):
        st.markdown(
            f"Confidence is how clear-cut the model's yes-or-no call is: how far the risk is "
            f"from {FORECAST_MIN:.0%}, where the forecast changes from No to Yes."
        )
        edges = (0, *CONFIDENCE_EDGES, 1)
        data_now = load_data()
        st.plotly_chart(
            charts.confidence_curve(
                data_now["DNA_Prob_Forecast"],
                data_now["Confidence_Score"],
                FORECAST_MIN,
                [
                    (edges[i], edges[i + 1], band, CONFIDENCE_COLOURS[band])
                    for i, band in enumerate(BANDS)
                ],
                [FORECAST_MIN, HIGH_RISK_MIN, 0.02],
            ),
            key="confidence_curve",
            width="stretch",
            config={"staticPlot": True},
        )
        st.caption(
            f"A {FORECAST_MIN:.0%} risk gives 0% confidence, because the model could go either "
            "way. The further the risk is from there, the more confident the model is, in either "
            "direction. So it is **Low for any risk between 8% and 43%**."
        )
    if selected_dna_risk != "All":
        table_data = table_data[table_data["Risk"] == selected_dna_risk]
    if selected_confidence != "All":
        table_data = table_data[table_data["Confidence"] == selected_confidence]

    # Highest risk first, so the first rows are the red ones worth clicking. Sorted here
    # rather than in the grid, so the selected position still matches table_data.
    table_data = table_data.sort_values(
        "DNA_Prob_Forecast", ascending=False, kind="stable"
    )
    if table_data.empty:
        st.info(
            "No appointments match the DNA Risk and Confidence chosen above. Choose **All** "
            "in either to see more."
        )
    else:
        table_key = (
            f"appointments_{filter_key}_{selected_dna_risk}_"
            f"{selected_confidence}_{st.session_state.get('table_nonce', 0)}"
        )
        # The table's selection from the click that caused this run, to shade that row
        picked = st.session_state.get(table_key)
        picked_rows = [r for r, _ in (picked.selection.cells if picked else [])]
        # Keyed so style.css can keep this table's scrollbar visible
        with st.container(key="appointments_grid"):
            table = st.dataframe(
                appointment_table(
                    table_data, highlight_row=picked_rows[0] if picked_rows else None
                ),
                column_config=APPOINTMENT_COLUMNS,
                hide_index=True,
                alt="Upcoming appointments with the forecast risk of not attending",
                on_select=open_explanation,
                # Cells only: row selection would add a column of tickboxes
                selection_mode="single-cell",
                # A fixed number of rows; it scrolls
                row_height=TABLE_ROW_HEIGHT,
                height=(TABLE_ROWS + 1) * TABLE_ROW_HEIGHT + 3,
                # A new key when any filter changes, so an old selection never points at a
                # different appointment, and when the explanation is dismissed, to clear it
                key=table_key,
            )
        # A clicked cell selects its row. Positions refer to the rows as passed in, whatever
        # sorting the user applies
        selected_rows = [r for r, _ in table.selection.cells if r < len(table_data)]
        # Only on the run the click caused, so it doesn't reopen on later reruns
        if st.session_state.pop("explain_open", False) and selected_rows:
            explain_appointment(
                table_data.iloc[selected_rows[0]],
                (filtered_data, filter_key, sidebar_choices),
            )


@st.fragment
def explainable_table(
    rows,
    filtered_data,
    filter_key,
    sidebar_choices,
    name,
    unique="",
    call_list=False,
    flag=None,
):
    """An appointments table (in the order passed in) where clicking a row opens the explanation
    overlay, like the All appointments table; a fragment, so only this part reruns. `name` keys
    its pieces for style.css and `unique` tells apart tables that share a name (a different day's
    list must not keep the last one's selection). `call_list` adds the time and patient name, and
    `flag` marks rows (a True per row) to highlight. `filtered_data` (the sidebar's filters) and
    `sidebar_choices` are for the overlay's Overall tab."""
    with st.container(key=f"click_cta_{name}"):
        st.info(
            "**Click a row** to see why the model gave that appointment its risk of not attending.",
            icon=":material/touch_app:",
        )
    table_key = f"{name}_{filter_key}_{unique}_{st.session_state.get('table_nonce', 0)}"
    # The table's selection from the click that caused this run, to shade that row
    picked = st.session_state.get(table_key)
    picked_rows = [r for r, _ in (picked.selection.cells if picked else [])]
    # Keyed so style.css can keep this table's scrollbar visible
    with st.container(key=f"{name}_grid"):
        table = st.dataframe(
            appointment_table(
                rows,
                call_list=call_list,
                highlight_row=picked_rows[0] if picked_rows else None,
                flag=flag,
            ),
            column_config=CALL_LIST_COLUMNS if call_list else APPOINTMENT_COLUMNS,
            hide_index=True,
            on_select=open_explanation,
            # Cells only: row selection would add a column of tickboxes
            selection_mode="single-cell",
            # A new key when the explanation is dismissed, to clear the selection
            key=table_key,
        )
    # A clicked cell selects its row
    selected_rows = [r for r, _ in table.selection.cells if r < len(rows)]
    # Only on the run the click caused, so it doesn't reopen on later reruns
    if st.session_state.pop("explain_open", False) and selected_rows:
        explain_appointment(
            rows.iloc[selected_rows[0]], (filtered_data, filter_key, sidebar_choices)
        )


def demo_notice_full(key):
    """The fuller version of the notice, beside the opening text of the first tab and the dashboard.

    Its poster and video open in pop-ups (show_poster, show_video). The notice is drawn
    on several tabs, so key keeps each tab's buttons apart.
    """
    # In a keyed container so style.css can make the notice's text smaller
    with st.container(key=f"demo_notice_{key}"):
        st.warning(
            "**DEMONSTRATION VERSION**\n\n"
            "This is a demonstration of a :red[**real tool**], currently being piloted by "
            "outpatient teams in a London trust, which predicts attendance for real upcoming appointments. "
            f"In a {pilot.SERVICE} pilot, phoning the patients it flagged as high risk cut missed "
            f"appointments among them from {pilot.DNA_BEFORE:.0%} to {pilot.DNA_AFTER:.0%}.\n\n"
            f"Here, all the data for training and display is synthetic, from a fictional trust "
            f"({TRUST_NAME}): no real patients or Trust data are displayed.",
            icon=":material/science:",
        )
    # One per line: the labels are too long to sit side by side in the narrow column
    with st.container(key=f"demo_links_{key}"):
        if st.button(
            ":material/image: Read about the impact of this HSMA project",
            key=f"poster_{key}",
        ):
            show_poster()
        if st.button(
            ":material/play_circle: Hear more directly from the associate",
            key=f"video_{key}",
        ):
            show_video()


def demo_notice_short():
    """A slimmed-down notice for the first tab's sidebar: what this is in one line, then the
    poster and video buttons side by side (keyed like demo_notice_full's, so style.css styles
    them the same)."""
    with st.container(key="demo_notice_intro"):
        st.warning(
            "**Demonstration version** of a :red[**real tool**] piloted in a London trust. "
            "All the data here is synthetic.",
            icon=":material/science:",
        )
    with st.container(key="demo_links_intro", horizontal=True):
        if st.button(":material/image: Poster", key="poster_intro", width="stretch"):
            show_poster()
        if st.button(
            ":material/play_circle: Video", key="video_intro", width="stretch"
        ):
            show_video()


def intro_columns(key, notice=demo_notice_full, widths=(0.65, 0.35)):
    """Columns for a tab's opening text (left) beside the demonstration notice (right).

    Returns the left column. The reflow_stack_ key stacks them on narrower screens (style.css).
    """
    if (
        notice is None
    ):  # the notice is elsewhere (the sidebar): the text takes the full width
        return st.container(key=f"reflow_stack_{key}")
    left, right = st.container(key=f"reflow_stack_{key}").columns(
        list(widths), gap="large"
    )
    with right:
        notice(key)
    return left


def go_to_tab(label):
    """Switch tab from a button; the new tab would otherwise open at the old scroll position.
    Ported from the geographic playground mini."""
    st.session_state.page = label
    st.session_state.scroll_to_top = True


# Everything the bonus game tab remembers about a visitor's go
GAME_KEYS = ("game_clues", "game_best", "game_last", "game_prev", "game_trained")


def reset_game():
    """Clear the game for the next person at the stand."""
    for key in GAME_KEYS:
        st.session_state.pop(key, None)
    st.session_state.scroll_to_top = True


def reset_costs():
    """Put every cost assumption in the sidebar back to its default: a widget with no value in
    the session state starts again from its default. All their keys start with cost_."""
    for key in [k for k in st.session_state if k.startswith("cost_")]:
        del st.session_state[key]


# Every filter on the demo dashboard tab: the sidebar's four and the table's two
FILTER_KEYS = (
    "filter_site",
    "filter_specialty",
    "filter_type",
    "filter_location",
    "filter_risk",
    "filter_confidence",
)


def clear_filters():
    """Put every appointment filter (the sidebar's and the table's) back to All. Set rather
    than deleted: the page sets these keys again on every run, and All is always an option."""
    for key in FILTER_KEYS:
        st.session_state[key] = "All"


def go_to_view(view):
    """From the bottom of one demo dashboard view to the next (the sidebar's switch)."""
    st.session_state.appt_view = view
    st.session_state.scroll_to_top = True


def next_step(key, heading, text, label, on_click, args=()):
    """The box at the foot of each demo dashboard view (and the first tab): one way on, so a visitor
    isn't asked to choose. Styled by style.css (view_next_*, view_next_btn_*)."""
    st.divider()
    with st.container(key=f"view_next_{key}"):
        st.markdown(f"### {heading}\n{text}")
        st.button(
            label,
            key=f"view_next_btn_{key}",
            on_click=on_click,
            args=args,
            icon=":material/arrow_forward:",
            type="primary",
        )


# Streamlit app
# The tabs come first so style.css can pin them to the top as a navbar; the app title
# is drawn at the left of that bar by style.css.
# Dropdown in place of the tab pills on narrower screens (shown by style.css)
add_mobile_nav()
# Keyed and rerun on change, so buttons can switch tab (go_to_tab), as in the geographic mini.
# Why a model helps comes first: a visitor who sees only one page sees what it does better than
# the obvious rule, and how it reaches a patient's risk. The game is a bonus for those who stay
# (style.css hides the sidebar on it), and the demo dashboard, the real tool's screens on
# synthetic data, is last for those who want to try it.
TAB_INTRO = "Why use a model?"
TAB_CALLS = "How many people should we phone?"
TAB_GAME = "Beat the model"
TAB_DASH = "Try a demo dashboard"
TAB_LABELS = [TAB_INTRO, TAB_CALLS, TAB_GAME, TAB_DASH]
tab_intro, tab_calls, tab_game, tab_dash = st.tabs(
    TAB_LABELS, key="page", on_change="rerun"
)
# The open tab. Before the first click the session state may not hold one yet: the first tab.
# A label from before the tabs were renamed goes back to the first.
page = st.session_state.get("page")
if page not in TAB_LABELS:
    page = TAB_INTRO
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
        fitted_image(
            "resources/NIHR_APPLIED_RESEARCH_COLLABORATION_Logo_Horizontal_RGB_WHITE.png"
        ),
        width="stretch",
        link="https://arc-sw.nihr.ac.uk/",
    )

# Which view the demo dashboard tab shows. Only offered on that tab; the tabs are
# keyed, so the page state says which one is open.
VIEW_ALL = "All appointments"
VIEW_SUMMARY = "Weekly summary"
VIEW_WEEK = "Next week's call list"
VIEW_ICONS = {
    VIEW_ALL: ":material/table_rows:",
    VIEW_SUMMARY: ":material/calendar_view_week:",
    VIEW_WEEK: ":material/call:",
}
DEFAULT_MEASURE = next(iter(weekly.MEASURES))  # the first: the strictest list
# Start on all appointments, so one button is always shown as chosen. Set through the session
# state rather than default=, as go_to_view sets it; set again on every run, as Streamlit
# forgets a widget's value on a run that doesn't draw it (the other tabs). A value that isn't
# one of the views (say, a label from before they were renamed) goes back to the first.
if st.session_state.get("appt_view") not in VIEW_ICONS:
    st.session_state["appt_view"] = VIEW_ALL
st.session_state["appt_view"] = st.session_state["appt_view"]
if page == TAB_DASH:
    st.sidebar.segmented_control(
        "Click the buttons below to change the view",
        list(VIEW_ICONS),
        key="appt_view",
        format_func=lambda v: f"{VIEW_ICONS[v]} {v}",
        required=True,  # clicking the chosen view again doesn't clear it
    )
view = st.session_state.get("appt_view") or VIEW_ALL
# The pointers that show what to click play again each time the visitor comes back to a tab,
# or to a view of the demo dashboard, not just once per page load
click_hint.replay_on_change(f"{page}|{view if page == TAB_DASH else ''}")

# Filters, only on the demo dashboard. The other tabs use all the upcoming appointments (the
# phone tab shows its own controls instead; all of them keep the cost assumptions). The choices
# made on the dashboard are kept, and still applied to it, while another tab is open.
show_filters = page == TAB_DASH


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

# Before the empty check below, so a combination that matches nothing can still be cleared.
# Greyed out while nothing is filtered.
if show_filters:
    st.sidebar.button(
        "Clear filters",
        key="clear_filters",
        on_click=clear_filters,
        icon=":material/filter_alt_off:",
        width="stretch",
        disabled=all(st.session_state.get(k, "All") == "All" for k in FILTER_KEYS),
        help="Show all upcoming appointments again.",
    )

if filtered_data.empty:  # DEMO: avoid crashing on empty filter combinations
    st.info("No appointments match the selected filters.")
    st.stop()

# The model's forecast (DNA Risk, Confidence) is filtered above the appointments table
# instead, for that table only: here it would fight the tab's own forecast cards, grid and
# call list. Its choices are kept while the table isn't drawn (Streamlit forgets a widget's
# value on a run that doesn't draw it).
for key in ("filter_risk", "filter_confidence"):
    if st.session_state.get(key) not in ["All", *BANDS]:
        st.session_state[key] = "All"
    st.session_state[key] = st.session_state[key]

# The How many people should we phone? tab's own controls, only on that tab. Filled in from
# that tab's code below, which works out the slider range; None on the other tabs.
call_controls = None
if page == TAB_CALLS:
    call_controls = st.sidebar.container()

# How many of those reached a call gets to attend: on the first tab and the phone tab, one setting
# for both. Set through the session state rather than value=, and set again on every run, as
# Streamlit forgets a widget's value on a run that doesn't draw it.
st.session_state.setdefault("call_effect", DEFAULT_CALL_EFFECT)
st.session_state["call_effect"] = st.session_state["call_effect"]
if page == TAB_INTRO:
    # The demonstration notice, at the top of the sidebar on the first tab, above the slider
    # (the other tabs draw it beside their opening text, see intro_columns)
    with st.sidebar:
        demo_notice_short()
        st.divider()
    st.sidebar.slider(key="call_effect", **CALL_EFFECT_SLIDER)
    st.sidebar.caption(
        f"The {pilot.SERVICE.lower()} pilot saw about {DEFAULT_CALL_EFFECT}%."
    )

# Cost assumptions: not filters, so they stay out of filter_key below and editing them
# never clears a table selection
st.sidebar.divider()
st.sidebar.subheader("Cost assumptions")
band5 = costs.admin_hourly_cost(**costs.BAND5)
# Minutes per call stays out of the expander: the one assumption most people will want to try
minutes_per_call = st.sidebar.number_input(
    "Minutes per call",
    min_value=0.5,
    value=float(costs.MINUTES_PER_CALL),
    step=0.5,
    format="%.1f",
    key="cost_minutes",
    help=(
        "On average, including unanswered attempts and noting the outcome. "
        "Costed assuming a band 5 admin makes the calls."
    ),
)
# Keyed so style.css can hide the advanced settings on the first tab; the widgets are still drawn
with (
    st.sidebar.container(key="cost_advanced"),
    st.expander(
        "Advanced settings for calls and missed appointment costs",
        icon=":material/payments:",
    ),
):
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

st.sidebar.button(
    "Reset to defaults",
    key="reset_costs",
    on_click=reset_costs,
    icon=":material/restart_alt:",
    width="stretch",
    help="Put every cost assumption back to its starting value.",
)

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
        )
    ).encode()
).hexdigest()[:10]

# The lowest predicted risk on each call list (weekly.MEASURES), for the savings box
MEASURE_RISK_MIN = {
    "High_Risk_DNA": HIGH_RISK_MIN,
    "DNA_Forecast": FORECAST_MIN,
}


def forecast_savings(called, risk_min, weeks=None, effect_pct=DEFAULT_CALL_EFFECT):
    """Predicted value of the appointments kept by phoning every appointment in `called` (all
    those at or above `risk_min`), if a call gets `effect_pct`% of those who would have
    missed to attend, with the same number of calls made by the simple rule and at random.
    Worked out as the phone tab does: calls reach a would-be DNA as often as they did in the
    past appointments at the same cut-off. None when nobody would be phoned. `weeks` adds a
    weekly figure to the help text."""
    n_calls = len(called)
    dna_cost = costs.cost_per_dna_reached(called) if n_calls else None
    if dna_cost is None:
        return None
    past = load_past()
    precision = past.loc[past["DNA_Prob_Forecast"] >= risk_min, "Did_Not_Attend"].mean()
    effect = effect_pct / 100
    # Where the share came from, for the help text
    effect_source = (
        f"what the {pilot.SERVICE.lower()} pilot saw, with no comparison group, so calls may "
        "change fewer people here"
        if effect_pct == DEFAULT_CALL_EFFECT
        else f"as set in the sidebar; the {pilot.SERVICE.lower()} pilot saw about "
        f"{DEFAULT_CALL_EFFECT}%"
    )
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

    # The same number of calls made another way, with the same effect and value per missed
    # appointment: by the simple rule (phone those who missed most before) and at random.
    # Like the model's figure, from the past appointments, phoning the same share of them.
    _, game_past, rule = load_comparison()
    share = float((past["DNA_Prob_Forecast"] >= risk_min).mean())
    rule_reached = n_calls * rule_game.precision_at_share(game_past, rule, share)
    random_reached = n_calls * past["Did_Not_Attend"].mean()

    def kept(n_reached):
        return n_reached * effect * dna_cost

    return {
        "n_calls": n_calls,
        "value": saving["value_avoided"],
        "call_cost": saving["call_cost"],
        "rule": kept(rule_reached),
        "random": kept(random_reached),
        "help": f"{n_calls:,} calls at £{cost_per_call:.2f} each: £{saving['call_cost']:,.0f}, "
        f"or {few(saving['staff_hours'])} hours of staff time, so about "
        f"£{saving['net']:,.0f} after the cost of the calls.\n\nIn past appointments, "
        f"{precision:.0%} of {who} were missed, so about {reached:,.0f} calls would reach "
        f"someone who would otherwise miss.\n\nIf {effect_pct}% of them attend instead "
        f"({effect_source}), that's about {few(reached * effect)} fewer missed "
        f"appointments, each valued at about £{dna_cost:,.0f}.{per_week}\n\nThis is the value "
        "of the clinic slots kept, not cash: a Trust may not be paid more when a patient "
        "attends.\n\nThe costs come from **Cost assumptions** in the sidebar.",
        "compare_help": f"In past appointments, the same share of calls would have reached "
        f"someone who would otherwise have missed {precision:.0%} of the time with the model's list, "
        f"{rule_reached / n_calls:.0%} phoning those who missed most before (the simple rule) "
        f"and {random_reached / n_calls:.0%} at random. Each is valued the same way, if "
        f"{effect_pct}% of those reached who would have missed attend instead.",
    }


def period_length(df):
    """The span of the appointments in `df`: their dates, how many whole weeks (rounded up;
    appointments run Monday to Friday) and the span in words ("5 days" or "6 weeks")."""
    dates = pd.to_datetime(df["Activity_Date"])
    days = (dates.max() - dates.min()).days + 1
    weeks = -(-days // 7)
    length = f"{days} day{'s' * (days != 1)}" if days <= 7 else f"{weeks} weeks"
    return dates, weeks, length


def section_title(question, note, **kwargs):
    """A short question as a section's title, with what it counts in a muted note on the same
    line (styled by style.css), so the titles cost no extra height."""
    st.markdown(
        # An en space before the note, not a margin, so a note that wraps lines up on the left
        f"#### {question}&ensp;<span class='title-note'>{note}</span>",
        unsafe_allow_html=True,
        **kwargs,
    )


def outcome_cols(n_calls):
    """How many markers wide the compact outcome chart's two columns are, for `n_calls` calls
    in 1,000. 84 in all, shared in line with the columns' sizes: the chart is then wide and
    short enough to fill its column and sit beside the cards. The limits keep each column at
    least as wide as its box (23 markers), so the markers fill it."""
    left_cols = min(61, max(23, round(84 * n_calls / 1000)))
    return left_cols, 84 - left_cols


def calling_right_people(out, list_name, note, key, previous=None, token=None):
    """ "Are we calling the right people?": the past appointments as four boxes (on the call list
    or not, missed or not) beside the two rate cards, for the calls in `out`
    (calls.call_outcomes). `list_name` says how the call list was made, for the caption; `key`
    keeps the phone tab's copy and the first tab's apart (and keys them for style.css).
    `previous`, the outcomes shown before, makes the markers that change box fly there from
    where they were (move_markers.py); `token` tells that script this chart from the one
    before it."""
    meta = load_model_meta()
    section_title("Are we calling the right people?", note)
    # Keyed so style.css puts the cards under the chart on narrower screens
    left, right = st.container(key=f"reflow_stack_icons_{key}").columns(
        [0.68, 0.32], gap="large"
    )
    with left:
        cols = outcome_cols(out["calls"])
        st.plotly_chart(
            charts.outcome_blocks(out, *cols, compact=True, meta=token),
            width="stretch",
            config={"displayModeBar": False},
            key=f"outcome_blocks_{key}",
        )
        if previous is not None:
            move_markers.animate_moves(
                move_markers.outcome_moves(
                    previous, out, outcome_cols(previous["calls"]), cols
                ),
                token,
                f"outcome_blocks_{key}",
            )
        st.caption(
            f'Each marker is one of 1,000 past appointments. "On the call list" is {list_name} (no '
            "calls were made); the boxes show what really happened. Counts are scaled from "
            f"{meta['n_past_test']:,} past appointments to 1,000, so they can be out by a few."
        )
    # Keyed so style.css makes these cards more compact, to sit beside the short chart
    with right.container(key=f"rates_{key}", gap="small"):
        # Short titles on the cards, with the explanation in a caption under each number
        with st.container(border=True, gap="small", key=f"rate_card_precision_{key}"):
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
        with st.container(border=True, gap="small", key=f"rate_card_recall_{key}"):
            st.metric(
                "Misses reached",
                f"{out['recall']:.0%}",
                icon=":material/person_search:",
                help="Also called recall or sensitivity.",
            )
            st.caption(
                f"{out['called_dna']} of the {out['dna_total']} patients who did not attend would get a "
                f"call; {out['not_called_dna']} would not."
            )


def choose_way(way):
    """A tile in the comparison box on the first tab was clicked: show its calls under the tiles.
    A different tile also counts a change, so the rate cards under the tiles flash (style.css)
    and the chart's markers move to their new boxes (move_markers.py), once."""
    before = st.session_state.get("landing_way", "model")
    if way != before:
        st.session_state.landing_flash = st.session_state.get("landing_flash", 0) + 1
        st.session_state.landing_move = before
    st.session_state.landing_way = way


@st.fragment
def compare_ways(savings, length):
    """The first tab's "Why bother with machine learning, rather than just phoning those who've missed before?" box: the same calls three
    ways, by the model, by the simple rule and at random. It leads with what the rule can't do
    (spot people who'd never missed before), and each tile reads on its own ("41 / of 100 calls /
    reached someone who would have missed"). Clicking a tile shows who its calls would reach,
    under the tiles. A fragment, so a click reruns just this box. `savings` is
    forecast_savings() for all the upcoming forecast DNAs, over `length`."""
    ways, _, _ = load_comparison()
    n_calls = ways["n_calls"]
    past_dna_rate = load_past()["Did_Not_Attend"].mean()
    chosen = st.session_state.get("landing_way")
    if chosen not in ("model", "rule", "random"):
        chosen = "model"
    with st.container(border=True, key="landing_compare"):
        ways_text, ways_numbers = st.container(key="reflow_stack_landing_ways").columns(
            [0.27, 0.73], gap="large", vertical_alignment="center"
        )
        with ways_text:
            st.markdown(
                "#### Why bother with machine learning, rather than just phoning those who've missed before?"
            )
            st.markdown(
                f"**{ways['never_missed_share'] * 10:.0f} in 10 missed appointments are by people "
                "who'd never missed before.** That simple rule alone can't spot them, but the model can."
            )
            # What the figures were tested on, and that the data is synthetic: the gist in the
            # label, the details in a popover (the sidebar notice also says it's synthetic). A
            # popover floats over the page, so opening it doesn't push the tiles beside it about
            with st.popover(
                f"Tested on {1000:,} past appointments (synthetic)",
                icon=":material/science:",
            ):
                # The counts from the build (build/train_and_explain.py splits the past
                # appointments: the model learns from one part and is tested on another)
                n_learnt = load_rule_game()[1]["n_train"]
                n_tested = load_model_meta()["n_past_test"]
                st.caption(
                    f"The model learnt from **{n_learnt:,} past appointments**. It's tested "
                    f"fairly, on **{n_tested:,} others it never saw** while learning, so it "
                    "can't just remember who missed: like an exam with questions the student "
                    "hasn't seen before.\n\n"
                    f"To make them easy to read, the figures here are scaled from those "
                    f"{n_tested:,} to {1000:,} appointments: {ways['n_missed']} in every "
                    f"{1000:,} ({past_dna_rate:.0%}) were missed.\n\n"
                    "**They are synthetic.** The factors reflect what the real model found "
                    "(long journeys, long waits between appointments, appointments after "
                    "treatment had finished), but not exactly: how much each one counts here "
                    "is made up."
                )
        # The question the tiles answer, right above them, and the same words under each
        # number ("46 / of 100 calls / reached someone who would have missed"), so a tile makes
        # sense to someone who reads nothing else
        reached = "reached someone who would have missed"
        ways_numbers.markdown(
            f"**If the team made {n_calls} calls, how many would reach a patient who would "
            "otherwise have missed their appointment?**"
        )
        # What the £ in each tile means, above them rather than under, so the chart a click
        # changes sits right under the tiles
        if savings is not None:
            ways_numbers.caption(
                f"**£ in each box:** the value of appointments kept over the next {length} by "
                f"phoning {savings['n_calls']:,} patients chosen that way (as many as the model "
                f"forecasts as DNAs), less £{savings['call_cost']:,.0f} of staff time for the "
                "calls. Clinic time kept, not cash.",
                help=savings["compare_help"],
            )
        # The same call to action as above the appointment tables ("Click" in yellow, style.css)
        with ways_numbers.container(key="click_cta_ways"):
            st.info(
                "**Click a box** to see who those calls would reach.",
                icon=":material/touch_app:",
            )
        # One tile per way: the number reached per 100 calls, then the same in £ over the
        # upcoming weeks (the value of the appointments kept by phoning every forecast DNA,
        # and by the same number of calls made the other two ways). The £ is after the cost
        # of the calls, which is the same all three ways and is also given once above the tiles.
        tiles = [
            (
                "model",
                "The model's list",
                ways["model"],
                ":material/smart_toy:",
                "value",
                (
                    f"Phoning the {n_calls} appointments with the highest predicted risk. The model "
                    "also gives every patient a risk, so the list fits the calls the team can make, "
                    "and the reasons behind each one (see the chart further down)."
                ),
            ),
            (
                "rule",
                "Missed most before",
                ways["rule"],
                ":material/history:",
                "rule",
                (
                    f"Phoning the {n_calls} patients who had missed the most appointments before: "
                    "the simple rule most people would choose. Patients who've never missed look "
                    "the same to it."
                ),
            ),
            (
                "random",
                "At random",
                round(ways["random"]),
                None,
                "random",
                (
                    f"Phoning {n_calls} appointments picked at random: the baseline, with no way "
                    "of choosing who to call."
                ),
            ),
        ]
        with ways_numbers.container(horizontal=True, key="landing_ways"):
            for key, label, value, icon, pounds, help_text in tiles:
                # The chosen tile's key ends _chosen, so style.css can pick it out
                with st.container(
                    border=True, key=f"ways_{key}{'_chosen' if key == chosen else ''}"
                ):
                    # Invisible, and stretched over the whole tile by style.css, so a click
                    # anywhere on the tile chooses it
                    st.button(
                        f"Show who {label.lower()} would call",
                        key=f"pick_way_{key}",
                        on_click=choose_way,
                        args=(key,),
                    )
                    st.metric(
                        label,
                        f"{value}",
                        f"of {n_calls} calls",
                        delta_color="off",
                        delta_arrow="off",
                        delta_description=reached,
                        icon=icon,
                        help=help_text,
                    )
                    if savings is not None:
                        # After the cost of the calls, so the minutes per call count too; a
                        # minus where the calls cost more than they keep. "After call costs" is
                        # left to the caption above the tiles, so this fits on fewer lines; a
                        # non-breaking space keeps "6 weeks" together
                        net = savings[pounds] - savings["call_cost"]
                        st.markdown(
                            f"**{'-' if net < 0 else ''}£{abs(net):,.0f}** kept over "
                            f"{length.replace(' ', chr(160))}"
                        )
        # The first time a visitor opens this tab, a pointer shows them clicking a box: one
        # that isn't chosen, so the click visibly would change something. After a pause, so
        # they've read the question first. Only once the tab is open (see show_click_hint)
        if page == TAB_INTRO:
            click_hint.show_click_hint(
                "ways", f"ways_{'rule' if chosen == 'model' else 'model'}", delay=1500
            )

        # Who the chosen tile's calls would reach, as on the phone tab. No divider above, so
        # the chart a click changes stays close to the tiles.
        list_names = {
            "model": "the model's list",
            "rule": "the patients who had missed the most appointments before",
            "random": "a random pick (the counts expected on average)",
        }
        tile_label = next(label for key, label, *_ in tiles if key == chosen)
        # After a click on a different tile, the rate cards' numbers flash yellow (style.css).
        # The key swaps between two names on each change, each with its own copy of the
        # animation, so it plays again every time; before any change, no flash. The chart's
        # markers that change box fly there instead, just after the click (popped, so a later
        # rerun doesn't replay it).
        flash = st.session_state.get("landing_flash", 0)
        moved_from = st.session_state.pop("landing_move", None)
        outcomes = load_way_outcomes()
        with st.container(
            key=f"landing_who_flash{flash % 2}" if flash else "landing_who"
        ):
            calling_right_people(
                outcomes[chosen],
                list_names[chosen],
                f"{tile_label.lower()}: {n_calls} calls in 1,000 past appointments",
                "landing",
                previous=outcomes.get(moved_from),
                token=flash,
            )


# The patients the first tab's explanation chart picks from: label, range in words, and the
# predicted risks covered (from, up to but not including). High is the high-risk flag; Medium
# is forecast as a DNA but not flagged; Low is the table's Low band.
LOW_RISK_MAX = 0.05
RISK_GROUPS = {
    "high": ("High risk", f"{HIGH_RISK_MIN:.0%} or more", HIGH_RISK_MIN, 1.01),
    "medium": (
        "Medium risk",
        f"{FORECAST_MIN:.0%} to {HIGH_RISK_MIN:.0%}",
        FORECAST_MIN,
        HIGH_RISK_MIN,
    ),
    "low": ("Low risk", f"under {LOW_RISK_MAX:.0%}", 0.0, LOW_RISK_MAX),
}


def pick_patient(group, seed=None):
    """Show a patient from a risk group (RISK_GROUPS) on the first tab's explanation chart: one
    of the upcoming appointments in that range, other than the one on show. A fixed `seed` for
    the first patient, so the page opens the same way for each visitor."""
    upcoming = load_data()
    _, _, low, high = RISK_GROUPS[group]
    risk = upcoming["DNA_Prob_Forecast"]
    ids = upcoming.loc[(risk >= low) & (risk < high), "row_id"].tolist()
    current = st.session_state.get("intro_patient")
    others = [i for i in ids if i != current] or ids
    st.session_state.intro_group = group
    st.session_state.intro_patient = random.Random(seed).choice(others)


def choose_patient(group):
    """A patient button on the first tab was clicked: show a new patient from `group`, and play
    their chart from empty once (grow_waterfall.py). Counted, so the script can tell the new
    chart from the old."""
    pick_patient(group)
    st.session_state.intro_grow = st.session_state.get("intro_grow", 0) + 1
    st.session_state.intro_grow_pending = True


@st.fragment
def explain_a_patient():
    """The first tab's explanation chart, always on show: one patient's waterfall, with buttons
    to swap between a high-, medium- and low-risk patient or show another from the same group,
    and the "How to read this chart" guide in a column beside it. A fragment, so the buttons
    rerun just this part."""
    if (
        st.session_state.get("intro_group") not in RISK_GROUPS
        or "intro_patient" not in st.session_state
    ):
        pick_patient("high", seed=0)
    group = st.session_state.intro_group
    upcoming = load_data()
    row = upcoming[upcoming["row_id"] == st.session_state.intro_patient].iloc[0]
    meta = load_model_meta()
    expl = explain.patient_explanation(
        row, load_shap().loc[row["row_id"]], meta["base_value_pp"]
    )

    st.markdown("#### Why does the model give a patient their risk?")
    st.markdown(
        "The model doesn't just give a risk: it shows which facts about the appointment pushed "
        "that risk up or down. **Pick a patient** to see what it relied on for them."
    )
    st.warning(
        "This is not a real patient, and the factors shown here should not be used for "
        "service planning.",
        icon=":material/science:",
    )
    # Big buttons (style.css): the chosen group filled, the others outlined
    with st.container(horizontal=True, key="intro_risk_buttons"):
        for g, (label, span, _, _) in RISK_GROUPS.items():
            st.button(
                f"{label} ({span})",
                key=f"intro_group_{g}",
                type="primary" if g == group else "secondary",
                on_click=choose_patient,
                args=(g,),
            )
        st.button(
            "Show a different patient",
            key="intro_shuffle",
            icon=":material/shuffle:",
            on_click=choose_patient,
            args=(group,),
            help=f"Another {RISK_GROUPS[group][0].lower()} patient.",
        )
    # The first time a visitor scrolls down to these buttons, a pointer shows them clicking
    # one that isn't chosen ("Medium risk" at first), so the click visibly would change the
    # chart. Only once the tab is open (see show_click_hint)
    if page == TAB_INTRO:
        click_hint.show_click_hint(
            "patients",
            f"intro_group_{next(g for g in RISK_GROUPS if g != group)}",
            inner="button",
            delay=600,
            bottom_margin=0.25,
        )

    # Keyed so style.css stacks the guide under the chart on narrower screens
    chart_col, guide_col = st.container(key="reflow_stack_intro_shap").columns(
        [0.62, 0.38], gap="large"
    )
    with chart_col:
        st.markdown(
            f"**Patient {row['MRN']}**: {row['Specialty']}, {row['Appointment Type'].lower()} "
            f"appointment on {row['Activity_Date']:%d/%m/%Y}. Predicted risk of not attending: "
            f"**{expl['fx']:.0f}%**"
        )
        # After a click, the chart plays from empty, once (popped, so a later rerun doesn't
        # replay it)
        patient_waterfall(
            expl,
            "intro_waterfall",
            token=st.session_state.get("intro_grow", 0),
            grow=st.session_state.pop("intro_grow_pending", False),
        )
    with guide_col.container(border=True, key="intro_guide"):
        st.markdown("**How to read this chart** (a made-up example)")
        how_to_read("intro_guide_chart", stacked=True)
    # Two boxes side by side, the same height (style.css), stacked on a phone: what the chart
    # means, and how good the model is
    with st.container(horizontal=True, key="intro_caveats"):
        st.info(
            "**A predicted risk is an estimate** based on similar appointments in past data, "
            f"not a forecast of what this one person will do. {bars_caveat()}"
        )
        st.info(model_accuracy("**How good is the model?**"))


# ---------------------------------------------------------------- tab 1: Why use a model? (the landing page)
# What the model does better than the obvious rule, and how it reaches one patient's risk. Uses all
# the upcoming appointments: the dashboard's filters are hidden here.
with tab_intro:
    with intro_columns("intro", notice=None):  # its notice is in the sidebar
        st.markdown(
            """
            ### Can we reduce missed appointments with Artificial Intelligence (AI)?

            A <span class="highlight">machine-learning model</span>, trained on past appointments, predicts
            each upcoming appointment's **risk of not attending** <span class="term">(a DNA)</span>, so teams
            can phone the patients most likely to miss. Its forecasts
            <span class="highlight">support staff judgement; they don't make decisions</span>.
            """,
            unsafe_allow_html=True,
        )

    # The value of the appointments kept by phoning every forecast DNA among all the upcoming
    # appointments, and by the same calls made by the simple rule and at random
    upcoming_all = data.assign(
        DNA_Cost=costs.appointment_costs(data, dna_costs, share_lost)
    )
    _, intro_weeks, intro_length = period_length(upcoming_all)
    compare_ways(
        forecast_savings(
            upcoming_all[upcoming_all["DNA_Forecast"] == 1],
            FORECAST_MIN,
            intro_weeks,
            effect_pct=st.session_state["call_effect"],  # the slider in the sidebar
        ),
        intro_length,
    )

    explain_a_patient()

    # ---- The way on: one button, to the phone tab
    next_step(
        "intro",
        "How many people should we phone?",
        "The model gives a risk, not a yes or no, so someone has to choose who gets a call. See "
        "how many calls are worth making.",
        "Plan the calls",
        go_to_tab,
        (TAB_CALLS,),
    )


# ---------------------------------------------------------------- tab 4: Try a demo dashboard
# The real tool's screens on synthetic data: one tab, three views picked by the switch at the
# top of the sidebar. Last on the bar, but drawn before the phone tab and the game, which must
# stay last in the script.
with tab_dash:
    if view == VIEW_ALL:
        # What really happened to past appointments the model did not train on
        past = load_past()
        forecast_recall, forecast_precision = catch_rates(FORECAST_MIN)
        high_recall, high_precision = catch_rates(HIGH_RISK_MIN)

        # Opening text beside the demonstration notice (stacked on narrower screens, see style.css),
        # with the forecast cards under it, so the table, the thing to click, starts high on the page
        with intro_columns("intro_dash"):
            st.markdown(
                "### Try a demo dashboard\n\n"
                "This is what an outpatient team would see: every upcoming appointment with the "
                "model's predicted risk of not attending, highest risk first. **Narrow it down with "
                "the filters in the sidebar**, and switch views at the top of the sidebar."
            )
            # Summary of the filtered appointments: three cards, each a metric and a caption,
            # stretched to the same height. The risk cut-offs are in the labels and the detail in
            # each card's help, to keep the row quiet.
            n_filtered = len(filtered_data)
            n_past_dnas = int(past["Did_Not_Attend"].sum())
            dates, weeks, length = period_length(filtered_data)
            n_forecast = int(filtered_data["DNA_Forecast"].sum())
            # The value of the appointments kept by phoning every forecast DNA
            savings = forecast_savings(
                filtered_data[filtered_data["DNA_Forecast"] == 1], FORECAST_MIN, weeks
            )

            def card_caption(count):
                return f"{count / n_filtered:.1%} of appointments"

            # Keyed so style.css can make the middle card wider than the two beside it
            with st.container(horizontal=True, key="forecast_cards"):
                with st.container(border=True, height="stretch"):
                    st.metric(
                        f"Forecast DNAs ({FORECAST_MIN:.0%}+ risk)",
                        f"{n_forecast:,}",
                        icon=":material/event_busy:",
                        help=f"Appointments the model forecasts as a DNA: a predicted risk of "
                        f"{FORECAST_MIN:.0%} or more (the 'Expected DNA' column). Not all of them will be "
                        f"missed, and some lower-risk appointments will be. Of the {n_past_dnas:,} "
                        "appointments missed in the past appointments the model was tested on, "
                        f"{forecast_recall:.0%} had a risk this high; of all those forecast as a DNA, "
                        f"{forecast_precision:.0%} were missed.",
                    )
                    st.caption(card_caption(n_forecast))
                with st.container(border=True, height="stretch"):
                    n_high = int(filtered_data["High_Risk_DNA"].sum())
                    st.metric(
                        f"Confident DNAs ({HIGH_RISK_MIN:.0%}+ risk)",
                        f"{n_high:,}",
                        icon=":material/warning:",
                        help=f"Appointments with a predicted risk of {HIGH_RISK_MIN:.0%} or more (the "
                        f"Very High band). Of the {n_past_dnas:,} appointments missed in the past "
                        f"appointments the model was tested on, {high_recall:.0%} had a risk this high; "
                        f"of all those flagged, {high_precision:.0%} were missed.",
                    )
                    st.caption(card_caption(n_high))
                    # Bold and full size, not caption: this is the card's takeaway
                    st.markdown(
                        "**If we can only call a few patients, calling these gives the best "
                        "return on staff time.**"
                    )
                with st.container(border=True, height="stretch"):
                    if savings is None:
                        st.metric(
                            "Value of appointments kept", "–", icon=":material/savings:"
                        )
                        st.caption(
                            "None of the appointments in your filters is forecast as a DNA, so "
                            "there are no calls to cost."
                        )
                    else:
                        st.metric(
                            f"Value of appointments kept ({length})",
                            f"£{savings['value']:,.0f}",
                            icon=":material/savings:",
                            help=savings["help"],
                        )
                        st.caption(
                            f"If we phone all {n_forecast:,} forecast DNAs, {dates.min():%d %b} to "
                            f"{dates.max():%d %b}, and {DEFAULT_CALL_EFFECT}% of those who'd have "
                            "missed attend instead."
                        )

        # The weekly grid has its own view (VIEW_SUMMARY), so the table shows every appointment
        # in the sidebar's filters
        appointments_table(
            filtered_data,
            filter_key,
            [
                ("site", selected_reporting_site),
                ("specialty", selected_specialty),
                ("appointment type", selected_appointment_type),
            ],
        )
        # The first time a visitor opens this view, a pointer shows them clicking a row. Only
        # once the tab is open (see show_click_hint)
        if page == TAB_DASH:
            click_hint.show_click_hint(
                "appointments",
                "appointments_grid",
                TABLE_ROW_HEIGHT,
                inner='[data-testid="stDataFrame"]',
            )

        # ---- The way on: one button, to the next view (style.css)
        next_step(
            "all",
            "See the weekly summary",
            "Which days have the most appointments the model expects to be missed, week by week.",
            "See the weekly summary",
            go_to_view,
            (VIEW_SUMMARY,),
        )

    elif view == VIEW_SUMMARY:
        # The forecast for every upcoming week, by weekday. Clicking a day lists its appointments
        # under the grid (the call list view covers just next week, ready to phone).
        with intro_columns("intro_summary"):
            st.markdown(
                "### Predicted non-attenders, week by week\n\n"
                "Which days have the most appointments the model expects to be missed? Each row is "
                "one week of the upcoming appointments in the sidebar's filters. For next week only, "
                "ready to phone, choose **Next week's call list** in the sidebar."
            )
            # In the intro column, beside the tall notice, so the grid starts higher
            measure = (
                st.segmented_control(
                    "Colour the grid by",
                    list(weekly.MEASURES),
                    default=DEFAULT_MEASURE,
                    key="grid_measure",
                    help="  \n".join(
                        f"**{name}:** {mm.help}" for name, mm in weekly.MEASURES.items()
                    ),
                )
                or DEFAULT_MEASURE
            )
            m = weekly.MEASURES[measure]
            column = m.column
            h = weekly.headline(filtered_data, column)
            low, high = m.thresholds
            st.info(
                "**Click a day in the table** to list that day's appointments below, highest risk "
                "first. Click it again to hide the list.",
                icon=":material/touch_app:",
            )
            # A key with a worked example and the colours, in one line above the grid
            st.markdown(
                f"""<div class="grid-key">
                <span><b>3 of 118 (3%)</b> = 118 appointments in total, 3 {m.cell_label}
                (3% of total)</span>
                <span><i class="sw" style="background:#198754"></i>up to {low:.0%}</span>
                <span><i class="sw" style="background:#e0a526"></i>{round(low * 100) + 1}% to {round(high * 100) - 1}%</span>
                <span><i class="sw" style="background:#dc3545"></i>{high:.0%} or more</span>                </div>""",
                unsafe_allow_html=True,
            )

        # Week-by-weekday table: earliest week first, weekdays with no appointments left out
        counts, sums, rates = weekly.weekly_tables(filtered_data, column)
        week_index = pd.Index(
            [weekly.week_label(w, year=True) for w in counts.index], name="Week ending"
        )
        cell_text = pd.DataFrame(
            {
                d: [
                    f"{s:.0f} of {c} ({r:.0%})" if c else "–"
                    for c, s, r in zip(counts[d], sums[d], rates[d])
                ]
                for d in counts.columns
            },
            index=week_index,
        )
        # Colour from the rate, opacity from the number of calls
        cell_styles = weekly.cell_styles(rates, sums, low, high).set_axis(week_index)

        week_table = st.dataframe(
            cell_text.style.apply(lambda _: cell_styles, axis=None),
            width="stretch",
            on_select="rerun",
            selection_mode="single-cell",
            key=f"weekly_{filter_key}",
        )
        # The overall summary ("Across 6 weeks, ...") sits under the grid it describes, as small
        # cards: plain bordered boxes, not metrics. Keyed so style.css shares the row equally.
        with st.container(horizontal=True, key="summary_cards"):
            for label, text in weekly.summary_cards(h, measure):
                with st.container(border=True, height="stretch"):
                    st.caption(label)
                    st.markdown(text)

        # The clicked day's appointments, highest risk first (clicks on the week labels are ignored)
        picked = [
            (r, day)
            for r, day in week_table.selection.cells
            if r < len(counts) and day in counts.columns
        ]
        if picked:
            r, day = picked[0]
            date = counts.index[r] - pd.Timedelta(days=6 - weekly.DAYS.index(day))
            day_data = filtered_data[
                pd.to_datetime(filtered_data["Activity_Date"]) == date
            ].sort_values("DNA_Prob_Forecast", ascending=False, kind="stable")
            st.subheader(
                f"{date:%A} {date.day} {date:%B}: {len(day_data)} appointments"
            )
            st.caption(
                f":yellow-badge[Yellow rows] are on the call list: those {m.day_phrase}."
            )
            explainable_table(
                day_data,
                filtered_data,
                filter_key,
                [
                    ("site", selected_reporting_site),
                    ("specialty", selected_specialty),
                    ("appointment type", selected_appointment_type),
                ],
                name="day_list",
                unique=f"{date:%Y%m%d}",
                flag=day_data[column] == 1,
            )

        # ---- The way on: one button, to the admin team's call list
        next_step(
            "summary",
            "See the list for the admin team",
            "What the team would actually work from: next week's appointments, earliest first, "
            "ready to phone from the top.",
            "See next week's call list",
            go_to_view,
            (VIEW_WEEK,),
        )

    else:
        with intro_columns("intro_weekly", widths=(0.70, 0.3)):
            st.title("Prioritised call list for admin team")

            st.markdown(
                "**Where to draw the line is a human decision.** "
                "The model gives every appointment a risk of not attending, not a yes or no."
                "\n\n- :material/arrow_upward: **Higher cut-off: fewer calls, better aimed.** "
                "Most reach people who would really have missed."
                "\n- :material/arrow_downward: **Lower cut-off: more misses caught, more calls.** "
                "Many reach people who would have come anyway."
                "\n\nThe right balance depends on how many calls the team can make, and how many "
                "real misses they reach."
            )
            # A call to action that holds its own control, so it can't be an st.info (which
            # can't contain widgets): a keyed container that style.css makes look like one
            # A choice kept from before "All appointments, ranked by risk" was removed
            if st.session_state.get("weekly_measure") not in weekly.MEASURES:
                st.session_state.pop("weekly_measure", None)
            with st.container(key="threshold_cta"):
                st.markdown(
                    ":material/touch_app: **Choose one of the two thresholds** to build the call "
                    "list, then explore it further in the **How many people should we phone?** tab."
                )
                measure = (
                    st.segmented_control(
                        "Who goes on the list",
                        list(weekly.MEASURES),
                        default=DEFAULT_MEASURE,
                        key="weekly_measure",
                        help="  \n".join(
                            f"**{name}:** {mm.help}"
                            for name, mm in weekly.MEASURES.items()
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
        # The call list covers the first week: the period's counted appointments
        period = next_week
        when = when_week
        listed = period[period[column] == 1]
        # Predicted value of phoning everyone on the list (None when nobody is on it)
        sv = forecast_savings(listed, MEASURE_RISK_MIN[column])

        # Each card in a bordered container stretched to the row's height, as on All
        # appointments, so the value card's extra lines don't leave the others short
        # Keyed so style.css can make the last (value) card wider than the other three
        with st.container(horizontal=True, key="week_cards"):
            with st.container(border=True, height="stretch"):
                st.metric(
                    f"Total appointments {when_week}",
                    f"{hw['n']:,}",
                    f"{hw['n'] - h['n'] / h['n_weeks']:+.0f} vs weekly average",
                    delta_color="off",
                    icon=":material/calendar_month:",
                    help=f"In the week ending {weekly.week_label(first_week, year=True)}, in the sidebar "
                    f"filters. The average over {h['n_weeks']} week{plural} is {h['n'] / h['n_weeks']:.0f}.",
                )
            vs_average = round(hw["total"] - h["weekly_average_total"])
            with st.container(border=True, height="stretch"):
                st.metric(
                    f"On the call list {when_week}",
                    f"{hw['total']:,.0f}",
                    f"{vs_average:+d} vs weekly average",
                    # No colour for no change (Streamlit would show "+0" in red)
                    delta_color="inverse" if vs_average else "off",
                    icon=":material/call:",
                    help=f"How many appointments {when_week} are on the list chosen above. This is not "
                    "how many will be missed: some of these patients would come anyway, and some "
                    f"lower-risk ones won't. {m.help} The weekly average is "
                    f"{h['weekly_average_total']:.0f} ({h['overall']:.0%} of all appointments).",
                )
                st.caption(f"{hw['overall']:.0%} of all appointments {when_week}")
            with st.container(border=True, height="stretch"):
                if sv is None:
                    st.metric(
                        f"Estimated value of appointments kept {when_week}",
                        "–",
                        icon=":material/savings:",
                    )
                    st.caption(
                        f"Nobody {when_week} is on this list, so there are no calls to cost."
                    )
                else:
                    st.metric(
                        f"Estimated value of appointments kept {when_week}",
                        f"£{sv['value']:,.0f}",
                        icon=":material/savings:",
                        help=sv["help"],
                    )
                    st.caption(
                        f"If we phone everyone on this list: {len(listed):,} calls in the week "
                        f"ending {weekly.week_label(first_week, year=True)}, if {DEFAULT_CALL_EFFECT}% "
                        "of those who'd have missed attend instead."
                    )
                    st.markdown(f":gray-badge[Staff time: £{sv['call_cost']:,.0f}]")

        # The call list itself, earliest appointment first: everyone on it is to be phoned, so
        # the soonest appointments come first (highest risk first at the same time)
        st.subheader("Calls to make")
        listed = listed.sort_values(
            ["Activity_Date", "Appointment_Time", "DNA_Prob_Forecast"],
            ascending=[True, True, False],
            kind="stable",
        )
        st.markdown(
            f"**{len(listed)} of {len(period)} appointments {when} are on the call list**: those "
            f"{m.day_phrase}, earliest appointment first. Phone from the top."
        )
        if period.empty:
            st.info("There are no appointments in the sidebar filters in this period.")
        elif listed.empty:
            st.info(f"None of the appointments {when} are {m.day_phrase}.")
        else:
            explainable_table(
                listed,
                filtered_data,
                filter_key,
                [
                    ("site", selected_reporting_site),
                    ("specialty", selected_specialty),
                    ("appointment type", selected_appointment_type),
                ],
                name="call_list",
                call_list=True,
            )
        # ---- The way on: one button, to the phone tab
        next_step(
            "calls",
            "Plan the calls",
            "How many calls should the team make, and would they be worth the staff time?",
            "Plan the calls",
            go_to_tab,
            (TAB_CALLS,),
        )


# ---------------------------------------------------------------- tab 2: How many people should we phone?
# Ported from the "Who would we phone?" page of lambda_ml_streamlit_exercise. The counts
# come from past appointments with known outcomes; the upcoming appointments (all of them:
# the sidebar hides the filters on this tab) only turn the chosen setting into calls per week.
with tab_calls:
    meta = load_model_meta()
    past = load_past()
    p_past = past["DNA_Prob_Forecast"].to_numpy()

    # The thought-experiment note and the sidebar prompt sit under the opening text and the notice
    # column is wider than on the other tabs, so the short opening text and the long notice come
    # out about as tall
    with intro_columns("intro_calls", widths=(0.55, 0.45)):
        st.subheader("How many people should we phone?")
        st.markdown(
            "This tab helps weigh up how many reminder calls to make, and who should get them.\n\n"
            "The model gives each appointment a **risk** of not attending, not a yes or no, so someone has "
            "to choose who gets a call."
        )
        st.caption(
            "**This is a thought experiment.** We don't know yet who will miss the upcoming appointments, so "
            f"the chart uses {meta['n_past_test']:,} **past** ones the model didn't learn from. Nobody in them "
            "was phoned, so we can check who the model would have picked against who really didn't attend."
        )
        st.info(
            "**Choose who gets a call in the sidebar**: by **Calls the team can make**, or by "
            "**Predicted risk**.",
            icon=":material/touch_app:",
        )

    # All the upcoming appointments, by week: the filters are hidden on this tab
    upcoming = data.assign(
        DNA_Cost=costs.appointment_costs(data, dna_costs, share_lost)
    )
    upcoming_week = pd.to_datetime(upcoming["Activity_Date"]).dt.to_period("W-SUN")
    per_week = upcoming_week.value_counts()
    n_weeks = len(per_week)

    # The calls slider runs up to the busiest upcoming week, rounded up to a multiple of 10.
    # Clamp the kept value first, so it is never out of range.
    busiest = max(10, int(-(-per_week.max() // 10) * 10))
    if st.session_state.get("calls_per_week", 0) > busiest:
        st.session_state["calls_per_week"] = busiest

    # Starting values set through the session state rather than default=/value=, so the
    # savings button on the appointments tab can preset them without a Streamlit warning
    # `or`, not setdefault: a session from before the control was required may hold None
    st.session_state["call_mode"] = (
        st.session_state.get("call_mode") or "Calls the team can make"
    )
    st.session_state.setdefault("calls_per_week", min(50, busiest))
    st.session_state.setdefault("risk_min", round(HIGH_RISK_MIN * 100))
    # Streamlit forgets a widget's value on a run that doesn't draw it: the controls are only
    # in the sidebar while this tab is open, and only one of the two sliders shows at a time.
    # (call_effect is kept further up, before the first tab can draw its slider.)
    for key in ("call_mode", "calls_per_week", "risk_min"):
        st.session_state[key] = st.session_state[key]

    def call_control(widget, key, *args, **kwargs):
        """A control in the sidebar while this tab is open; on the other tabs, its kept value."""
        if call_controls is None:
            return st.session_state[key]
        return getattr(call_controls, widget)(*args, key=key, **kwargs)

    mode = (
        call_control(
            "segmented_control",
            "call_mode",
            "Choose who to phone by",
            ["Calls the team can make", "Predicted risk"],
            required=True,  # clicking the chosen option again doesn't clear it
        )
        or "Calls the team can make"
    )

    if mode == "Calls the team can make":
        cap = call_control(
            "slider",
            "calls_per_week",
            "Calls the team can make per week",
            0,
            busiest,
            step=10,
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
            "who would have come anyway. **Move the slider in the sidebar and watch it change.**"
        )
    else:
        # Risk only: confidence is a function of the risk alone, so it adds nothing a risk bar doesn't, and
        # on its own it would also pick the patients the model is confident will attend.
        # Few risks pass 95% at an 8% DNA rate, so the slider stops there, in 1% steps
        risk_min = (
            call_control(
                "slider",
                "risk_min",
                "Phone if the predicted risk is at least",
                0,
                95,
                step=1,
                format="%d%%",
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
            f"app's high-risk flag, the patients phoned in the pilot; {FORECAST_MIN:.0%} phones everyone "
            "forecast as a DNA."
        )

    if call_controls is not None:
        call_controls.divider()
    effect = call_control("slider", "call_effect", **CALL_EFFECT_SLIDER)
    if call_controls is not None:
        call_controls.caption(
            f"The {pilot.SERVICE.lower()} pilot saw about {DEFAULT_CALL_EFFECT}%."
        )

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
        # Missed appointments a week the calls must save to cover their cost. They can only save
        # the ones they reach, so past that no effect could pay for them.
        be = econ["break_even_saved"]
        be = float("inf") if be is None else be
        can_break_even = be <= weekly_reached

        def appointments(n):
            """A weekly count of appointments: one decimal place under 10."""
            return f"{n:.1f}" if n < 10 else f"{n:.0f}"

    # The cut-off and the costs side by side, so the results fit on one screen. Keyed so
    # style.css stacks them on narrower screens.
    summary_left, summary_right = st.container(
        key="reflow_stack_phone_summary"
    ).columns([0.28, 0.72], gap="small")

    # Keyed so style.css gives it smaller numbers. The two cards are stacked, not side by side:
    # "about 100" is cut short in half of this narrow box
    with summary_left.container(border=True, height="stretch", key="phone_who"):
        # Per week, like the slider and the costs beside it: the one switch to per 1,000 past
        # appointments comes with the chart below, which says how the two connect
        section_title("Who are we calling?", "per week")
        share_called = upcoming_called.mean()
        st.metric(
            "Calls",
            f"about {weekly_calls:.0f}" if weekly_calls > 0 else "0",
            # The share links this to the 1,000 past appointments below; neutral, not good or bad
            # One decimal place under 1%, so a handful of calls doesn't read as 0%
            f"{share_called:.{1 if share_called < 0.01 else 0}%} of appointments"
            if weekly_calls > 0
            else None,
            delta_color="off",
            delta_arrow="off",
            help=f"On average over the {n_weeks} upcoming weeks. Quieter weeks have fewer "
            "appointments to call, so this can be a little under the slider.",
        )
        st.metric(
            "Cut-off risk"
            if mode == "Calls the team can make"
            else "Lowest risk phoned",
            f"{out['cutoff']:.0%}" if out["cutoff"] is not None else "nobody",
            help=how,
        )
        st.caption(
            "The cut-off follows from how many calls the team can make. Move the slider in the "
            "sidebar to change it."
            if mode == "Calls the team can make"
            else "You set the bar in the sidebar; the number of calls follows from it."
        )

    # Cost of the chosen setting; the assumptions and caveats sit in the cards' help
    if has_costs:
        with summary_right.container(border=True, height="stretch"):
            section_title(
                "What are we saving?",
                "per week, for all the upcoming appointments",
                help="Estimates from the sidebar's cost assumptions. A missed appointment's cost is the "
                "value of a clinic slot, not cash that comes back. The call cost is staff time only.",
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
            # The two pound figures, then the two counts of appointments, so the break-even sits
            # beside the number the calls might save
            c2.metric(
                "Possibly saved, after costs",
                # After the cost of the calls is the main figure, with a minus for a loss; the
                # figure before costs is the small one under it, neutral (no colour, no arrow)
                f"{'-' if econ['net'] < 0 else ''}£{abs(econ['net']):,.0f}",
                f"£{econ['value_avoided']:,.0f} before costs",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
                icon=":material/savings:",
                help=f"The value of the missed appointments the calls might save, less the cost of "
                f"the calls. If a call gets {effect}% of the people it reaches, who would have "
                f"missed, to attend. A missed appointment among the patients these calls reach is "
                f"valued at about £{dna_cost:,.0f} on average. Change this and the call cost under "
                "**Cost assumptions** in the sidebar.",
            )
            # The words in the label, not the value: a longer value is cut short on the card.
            # The labels can wrap onto two lines; style.css keeps the numbers level.
            c3.metric(
                "Appointments needing to be saved to break even on cost of calls",
                appointments(be) if can_break_even else "too many",
                border=True,
                height="stretch",
                icon=":material/balance:",
                help=(
                    f"The missed appointments a week the calls must save to cover their staff time: "
                    f"£{econ['call_cost']:,.0f} of calls, at about £{dna_cost:,.0f} a missed appointment. "
                    "It doesn't depend on the effect slider."
                    if can_break_even
                    else f"The calls would need to save {appointments(be)} missed appointments a week to "
                    f"cover their staff time, but they only reach about {appointments(weekly_reached)} "
                    "people who would miss."
                ),
            )
            c4.metric(
                "Appointments expected to be saved",
                appointments(weekly_avoided),
                # Neutral, for scale: how many appointments a week the saved ones come from
                f"of about {len(upcoming) / n_weeks:,.0f} a week",
                delta_color="off",
                delta_arrow="off",
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
            # The cards above give the verdict (the pill under "Possibly saved"), so this only sets
            # them side by side with the pilot: appointments saved per 3 calls, and early
            # cancellations if calls free slots as often as they did there. The pilot itself is
            # explained at the bottom of the tab, so it is named here.
            st.markdown(
                f"These calls are estimated to save **{3 * weekly_avoided / weekly_calls:.1f} "
                f"appointments per 3 calls**; a real-service pilot saved "
                f"{3 * pilot.SAVED_PER_CALL:.0f}.\n\nAt that pilot's early-cancellation rate, about "
                f"{weekly_calls * pilot.CANCEL_RATE:.0f} calls a week would free a slot for another "
                "patient."
            )
    elif out["calls"] > 0:
        summary_right.info(
            "None of the upcoming appointments would get a call with this setting, so there is "
            "nothing to cost. Lower the bar."
        )

    if out["calls"] == 0:
        st.info(
            "With this setting nobody would get a call. Raise the number of calls or lower the bar."
        )
        # No chart now, so the next one has nothing to move from
        st.session_state.phone_shown = None
    else:
        # When the setting changes, the markers that change box fly there (move_markers.py).
        # Against the outcomes the chart last showed, so it plays once per change; counted, so
        # the script can tell the new chart from the old.
        shown = st.session_state.get("phone_shown")
        counts = {k: out[k] for k, _ in calls.GROUPS} | {"calls": out["calls"]}
        moved = shown is not None and shown != counts
        if moved:
            st.session_state.phone_move = st.session_state.get("phone_move", 0) + 1
        st.session_state.phone_shown = counts
        # The rate cards' numbers flash yellow on a change too (style.css), as on the first
        # tab: the key swaps between two names on each change, each with its own copy of the
        # animation, so it plays again every time; before any change, no flash.
        flash = st.session_state.get("phone_move", 0)
        with st.container(
            key=f"phone_calling_flash{flash % 2}" if flash else "phone_calling"
        ):
            calling_right_people(
                out,
                "the model's list",
                # The bridge from the per-week figures above to the past appointments below
                f"the same setting replayed on 1,000 past appointments: {out['calls']:,} calls",
                "phone",
                previous=shown if moved else None,
                token=flash,
            )

    # What happened when the real tool was piloted (pilot.py): the evidence behind the effect
    # slider's starting value. Full width at the bottom, in an expander, so the results above
    # stay on one screen. Keyed so style.css shows the cards two per line on narrower screens.
    with st.expander("What happened in the real pilot?"):
        st.markdown(
            f"The {pilot.SERVICE} service phoned patients the live model flagged as high risk, from "
            f"{pilot.PERIOD}, to confirm, rearrange or reassure."
        )
        p1, p2, p3, p4 = st.container(key="reflow_halves_pilot").columns(4, gap="small")
        p1.metric(
            "Patients phoned",
            f"{pilot.CALLS}",
            border=True,
            height="stretch",
            icon=":material/call:",
            help="Patients the live model flagged as at high risk of not attending.",
        )
        p2.metric(
            "Missed after a call",
            f"{pilot.DNA_AFTER:.0%}",
            f"down from {pilot.DNA_BEFORE:.0%}",
            delta_arrow="off",
            border=True,
            height="stretch",
            icon=":material/event_available:",
            help=f"These patients typically missed {pilot.DNA_BEFORE:.0%} of their appointments. "
            f"Those who answered and said they would come missed only {pilot.DNA_CONFIRMED:.0%}, "
            "about the same as patients at low risk.",
        )
        p3.metric(
            "Extra appointments attended",
            f"{pilot.EXTRA_ATTENDED}",
            f"about £{pilot.REVENUE:,.0f} not lost",
            delta_arrow="off",
            border=True,
            height="stretch",
            icon=":material/savings:",
            help=f"Over {pilot.WEEKS} weeks: about £{pilot.VALUE_PER_APPOINTMENT:,.0f} an appointment, "
            "in revenue the trust would otherwise have lost.",
        )
        p4.metric(
            "Appointments saved per call",
            "2 in every 3",
            border=True,
            height="stretch",
            icon=":material/balance:",
            help=f"{pilot.EXTRA_ATTENDED} extra appointments from {pilot.CALLS} calls.",
        )
        # The pilot's limits are in the warning below, so the caption keeps only the extra finding
        st.caption(
            f"The calls also brought {pilot.EARLY_CANCELLATIONS} early cancellations, so those slots could be "
            "offered to other patients."
        )
        st.warning(
            f"**The {pilot.SERVICE.lower()} pilot suggests calls help a lot, but it isn't proof.** The "
            "chart above shows who the calls would *reach*, not missed appointments they would *prevent*.\n\n"
            f"- Even in the pilot, {pilot.DNA_AFTER:.0%} of the patients phoned still missed.\n"
            "- The pilot compared patients with how often they usually missed, not with similar patients who "
            "weren't phoned at the same time, so some of the fall might have happened anyway.\n"
            "- It phoned patients at very high risk in one service. Calls lower down the list, or in other "
            "services, may change fewer people.\n"
            "- Calls to people who would have come anyway still take staff time, and could worry them.\n"
            "- The surest way to know is to test it: for example, randomly phone half of the flagged patients "
            "and compare how many from each half attend.",
            icon=":material/warning:",
        )

    # ---- The way on: one button, to the game
    next_step(
        "game",
        TAB_GAME,
        "Make your own rule for who to phone, and see if you can catch more missed appointments "
        "than a machine learning model.",
        "Play the game",
        go_to_tab,
        (TAB_GAME,),
    )


# ---------------------------------------------------------------- tab 3: Beat the model
# First on the bar but last in the script: the training animation pauses between frames, so
# the other tabs are drawn before it starts. The visitor builds a rule from the clues the model
# sees, each with an equal say; then a machine learning model learns its own weighting from more and more
# past appointments. Everything is scored on the same past appointments as the phone tab, scaled to
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
        ":material/science: Demonstration version: all the appointments here are synthetic, from a "
        f"fictional trust ({TRUST_NAME}). The "
        "clues reflect what the real model found, but not exactly: how much each one counts is made "
        "up. No real patients or Trust data are displayed."
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
                "- **Patient A** has missed 3 appointments before; only 7% of patients have missed more, "
                "so A scores 0.93. A is 70, younger than only about 14% of patients: 0.15. "
                "**Total 1.08.**\n"
                "- **Patient B** has never missed one. About 70% of patients haven't, and they "
                "share the bottom of that ranking: 0.35. B is 25, younger than about 92% of patients: 0.93. "
                "**Total 1.28.**\n\n"
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
        # No chart now, so the next one has nothing to move from
        st.session_state.game_shown = None
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
        # When the rule changes, the markers that change box fly there (move_markers.py).
        # Against the outcomes the chart last showed, so it plays once per change; counted, so
        # the script can tell the new chart from the old.
        shown = st.session_state.get("game_shown")
        moved = shown is not None and shown != mine["outcomes"]
        if moved:
            st.session_state.game_move = st.session_state.get("game_move", 0) + 1
        st.session_state.game_shown = mine["outcomes"]
        game_token = st.session_state.get("game_move", 0)
        # 50 wide on the right so the chart is wide and short enough to fill its column
        game_cols, game_heading = (10, 50), "Using your rule to rank the list"
        with left:
            st.plotly_chart(
                charts.outcome_blocks(
                    mine["outcomes"], *game_cols, heading=game_heading, meta=game_token
                ),
                width="stretch",
                config={"displayModeBar": False},
                key="game_rule_icons",
            )
            if moved:
                move_markers.animate_moves(
                    move_markers.outcome_moves(
                        shown,
                        mine["outcomes"],
                        game_cols,
                        game_cols,
                        game_heading,
                        compact=False,
                    ),
                    game_token,
                    "game_rule_icons",
                    compact=False,
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
                f"#{rule_game.rank_of(caught, all_catches)} of {n_rules:,}",
                border=True,
                icon=":material/leaderboard:",
                help=f"Out of all {n_rules:,} rules you can make with these buttons, scored the same way.",
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
                # Teal and bold (style.css), and up for longer, so it isn't missed mid-race
                st.toast(
                    "The model just overtook your best rule!",
                    icon=":material/emoji_events:",
                    duration="long",
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
        final = frames[-1]
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
        # The best rule the buttons can make, named, so the verdict is honest about how close
        # it gets (the same comparison as "Why bother with machine learning, rather than just phoning those who've missed before?" on the first tab)
        best_rule = max(all_catches, key=lambda rule: all_catches[rule])
        names = [f"*{rule_game.CLUES[c].lower()}*" for c in best_rule]
        best_rule_text = (
            names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        )
        found = (
            "You found the best rule the buttons can make, and the model still won."
            if yours >= best_possible
            else f"No rule from the buttons can match it: the best, {best_rule_text}"
            f"{' on its own' if len(best_rule) == 1 else ''}, reaches {best_possible} of {n_missed}."
        )
        # How far ahead the model is, in words that stay true if the data changes: a simple
        # rule can do most of the work when the model is only a few ahead of the best one
        lead = theirs - best_possible
        verdict = (
            "**So a simple rule does most of the work.** The model adds the last few, plus a risk "
            "for every patient (to size the call list) and the reasons behind it."
            if lead <= 3
            else f"**The model reaches {lead} more than any rule from the buttons**, by learning "
            "how much each clue should count. It also gives a risk for every patient (to size the "
            "call list) and the reasons behind it."
        )
        equal_say = (
            "Your rule gives every clue an equal say: tuning the weights by hand is hard."
            if len(best["clues"]) > 1
            else "Your rule uses one clue on its own."
        )

        # The model's card wider: it has more to say than the short "nobody gets them all" card
        c1, c2 = st.container(key="reflow_stack_game_verdict_cards").columns(
            [0.62, 0.38], gap="small"
        )
        with c1.container(border=True, key="title_card_game_say", height="stretch"):
            st.markdown(
                ":material/balance: **Same clues, but the model can learn which matter most as it goes.**"
            )
            st.markdown(
                f"{equal_say} The model gives "
                f"*{rule_game.CLUE_SHORT[top_two[0]].lower()}* {say[top_two[0]]:.0%} and "
                f"*{rule_game.CLUE_SHORT[top_two[1]].lower()}* {say[top_two[1]]:.0%}, and the rest less."
                f"\n\n{found}\n\n{verdict}"
            )
        with c2.container(border=True, key="title_card_game_limit", height="stretch"):
            st.markdown(
                ":material/visibility_off: **Nobody – not even the best computer model – gets "
                "them all.**"
            )
            # The weather forecast as the everyday picture of a risk, not a certainty
            st.markdown(
                "A 70% chance of rain doesn't mean it will rain. Missed appointments work the "
                "same way: some high-risk patients turn up, and some low-risk ones miss because "
                "of <span class='highlight'>things no data records</span>, like a bus delay or a "
                "poorly child.",
                unsafe_allow_html=True,
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
        # The two points picked out in the app's yellow highlight (style.css), as in the intro
        w2.markdown(
            "- Computers can <span class='highlight'>re-learn when things change</span>, like a new "
            "site or new booking habits. "
            "\n- Some models can also <span class='highlight'>spot clues that matter more in "
            "combination</span>, like a long gap mattering more before a first appointment, or "
            "Monday mornings being worse than other days.",
            unsafe_allow_html=True,
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
            "### Try a demo dashboard\n"
            "The real tool's screens, on synthetic appointments: every upcoming appointment with its "
            "predicted risk, which ones it flags for staff to call, and why."
        )
        onward.button(
            "Try a demo dashboard",
            key="to_dash",
            on_click=go_to_tab,
            args=(TAB_DASH,),
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
            args=(TAB_CALLS,),
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
