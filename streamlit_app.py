import base64
import hashlib
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib
from matplotlib.colors import to_hex

import charts
import explain
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
tab_appts, tab_weekly = st.tabs(["Appointments", "Weekly summary"])
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

# ---------------------------------------------------------------- tab 1
with tab_appts:
    demo_notice()

    # Display filtered data
    st.write(
        "Filtered Data Count:",
        len(filtered_data),
        "      |       Forecasted DNAs:",
        sum(filtered_data["DNA_Forecast"]),
        "      |       DNA Rate - (High Confidence & High Risk) %:",
        round(sum(filtered_data["High_Risk_DNA"]) / len(filtered_data) * 100, 2),
    )

    # Fields to display
    fields_to_display = [
        "Activity_Date",
        "MRN",
        "Mobile Telephone",
        "Specialty",
        "Appointment Location Text",
        "Appointment Type",
        "DNA_Forecast_Text",
        "Confidence",
        "Probabilty",
    ]

    display_df = filtered_data[fields_to_display]

    filtered_data_renamed = display_df.rename(
        columns={
            "Activity_Date": "Appointment Date",
            "MRN": "MRN",
            "Mobile Telephone": "Mobile Number",
            "Specialty": "Specialty",
            "Appointment Location Text": "Location",
            "Appointment Type": "Appointment Type",
            "DNA_Forecast_Text": "Expected DNA",
        }
    )

    st.caption("Click a row to see why the model gave that appointment its risk.")

    # A new key whenever the filters change, so an old selection never points at
    # a different appointment in the newly filtered table.
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

    # Display filtered data
    table = st.dataframe(
        filtered_data_renamed,
        column_config={
            "Appointment Date": st.column_config.DateColumn(
                "Appointment Date",
                format="DD/MM/YYYY",
            ),
            "MRN": st.column_config.TextColumn(),
            "Mobile Number": st.column_config.TextColumn(),
            #'Mobile Number (GP Record)': st.column_config.TextColumn(),
            "Specialty": st.column_config.TextColumn(),
            "Location": st.column_config.TextColumn(),
            "Appointment Type": st.column_config.TextColumn(),
            "Expected DNA": st.column_config.TextColumn(),
            "Confidence": st.column_config.TextColumn(),
            "Probabilty": st.column_config.NumberColumn(
                "DNA Risk", format="%.2f%%", help="Probability of DNA"
            ),
        },
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"appointments_{filter_key}",
    )

    # Selected positions refer to the rows as passed in, whatever sorting the user applies
    selected_rows = [r for r in table.selection.rows if r < len(filtered_data)]
    if not selected_rows:
        st.info(
            "**Click a row in the table** to see what made the model give that "
            "appointment its risk of not attending.",
            icon=":material/touch_app:",
        )
    else:
        row = filtered_data.iloc[selected_rows[0]]
        meta = load_model_meta()
        expl = explain.patient_explanation(
            row, load_shap().loc[row["row_id"]], meta["base_value_pp"]
        )

        st.subheader(
            f"Why {row['MRN']} on {row['Activity_Date']:%d/%m/%Y} has a "
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
    st.markdown(
        "Appointments | DNAs (Forecast DNA Rate %). Below 8% = Green over 20% = Red"
    )

    # Prevent errors if filters result in no data
    if not filtered_data.empty:
        # Create a copy to avoid SettingWithCopyWarning
        pivot_df = filtered_data.copy()

        # Ensure the Activity_Date column is a datetime type
        pivot_df["Activity_Date"] = pd.to_datetime(pivot_df["Activity_Date"])

        # Create 'Day of Week' and 'Weekending' (Sunday) columns for pivoting
        pivot_df["Day of Week"] = pivot_df["Activity_Date"].dt.day_name()
        pivot_df["Weekending"] = (
            pivot_df["Activity_Date"]
            .dt.to_period("W-SUN")
            .apply(lambda r: r.end_time)
            .dt.strftime("%d-%b-%Y")
        )

        # Define the correct order for the days of the week
        days_of_week = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        pivot_df["Day of Week"] = pd.Categorical(
            pivot_df["Day of Week"], categories=days_of_week, ordered=True
        )

        # Create the pivot table
        pivot_table = pd.pivot_table(
            pivot_df,
            values="High_Risk_DNA",
            index="Weekending",
            columns="Day of Week",
            aggfunc=["count", "sum"],
        )

        # Extract the 'count' (total appointments) and 'sum' (DNA count) tables
        counts = (
            pivot_table.get("count", pd.DataFrame())
            .reindex(columns=days_of_week, fill_value=0)
            .fillna(0)
            .astype(int)
        )
        sums = (
            pivot_table.get("sum", pd.DataFrame())
            .reindex(columns=days_of_week, fill_value=0)
            .fillna(0)
            .astype(int)
        )

        # Calculate DNA rate, handling division by zero
        dna_rate = (sums / counts).fillna(0)

        # --- Styling Logic ---
        def style_dna_rate(val):
            cmap = matplotlib.colormaps["RdYlGn_r"]
            norm = matplotlib.colors.Normalize(vmin=0.08, vmax=0.20)

            font_color = "black"  # Default font color
            bg_color = "white"  # Default background color

            if pd.isna(val) or val == 0:
                return f"background-color: {bg_color}; color: {font_color};"

            # Condition for dark green background where white text is needed
            if val < 0.08:
                bg_color = "#198754"  # A darker, high-contrast green
                font_color = "white"
            # *** CHANGE HERE: Use dark red for high rates and set font to white ***
            elif val > 0.20:
                bg_color = "#dc3545"  # A darker, high-contrast red
                font_color = "white"
            else:
                bg_color = to_hex(cmap(norm(val)))

            return f"background-color: {bg_color}; color: {font_color};"

        # Create a DataFrame of styles by applying the function to the DNA rate data
        style_df = (
            dna_rate.map(style_dna_rate)
            if hasattr(dna_rate, "map")
            else dna_rate.applymap(style_dna_rate)
        )

        # Create the text to display in the table using the new format
        # Using a loop for clarity, performance is fine for this small table
        formatted_table = pd.DataFrame(
            "", index=counts.index, columns=counts.columns
        )
        for r in formatted_table.index:
            for c in formatted_table.columns:
                count = counts.loc[r, c]
                dna_sum = sums.loc[r, c]
                rate = dna_rate.loc[r, c]
                if count > 0:
                    formatted_table.loc[r, c] = f"{count} | {dna_sum} ({rate:.0%})"
                else:
                    formatted_table.loc[r, c] = "0 | 0 (0%)"

        # Sort index to have the most recent week at the top
        formatted_table = formatted_table.sort_index(
            key=lambda idx: pd.to_datetime(idx, format="%d-%b-%Y"),
            ascending=False,
        )
        # Ensure style df matches the sorted index
        style_df = style_df.loc[formatted_table.index]

        # Apply the styles to the formatted text table
        styler = formatted_table.style.apply(lambda x: style_df, axis=None)

        st.dataframe(styler, width="stretch")
    else:
        st.info("No appointment data available for the selected filters.")
    # --- END of finalized visualization ---
