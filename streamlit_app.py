import streamlit as st
import pandas as pd
import matplotlib
from matplotlib.colors import to_hex

from demo_data import generate_forecast  # DEMO: synthetic data replaces SQL

st.set_page_config(page_title="Outpatient DNA Forecast", layout="wide")

# NHS styling
st.markdown(
    """
    <style>
    /* NHS Blue */
    .main {
        background-color: #f0f4f5;
    }
    header[data-testid="stHeader"] {
        background: #005eb8;
    }
    .stApp {
        font-family: 'Arial', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #005eb8;
        font-family: 'Arial', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .css-1d391kg {  /* Sidebar header */
        background-color: #005eb8 !important;
        color: #fff !important;
    }
    .css-1v0mbdj {  /* Sidebar */
        background-color: #e8edee !important;
    }
    .stButton>button {
        background-color: #007f3b;
        color: white;
        border-radius: 4px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #005eb8;
        color: white;
    }
    /* Increase sidebar width */
    section[data-testid="stSidebar"] {
        min-width: 350px;
        max-width: 350px;
        width: 350px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# NHS logo (optional)
st.markdown(
    """
    <div style="position: absolute; top: 1.2rem; right: 2rem; z-index: 100;">
        <img src="https://assets.nhs.uk/images/nhs-logo.png" width="120">
    </div>
    """,
    unsafe_allow_html=True,
)


# DEMO: data comes from demo_data.py instead of vw_Combined_Future_OP_Forecast
@st.cache_data
def load_data():
    return generate_forecast()


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


# Streamlit app
st.title("Outpatient DNA Forecast")
st.warning(
    "DEMONSTRATION VERSION - all data shown is synthetic. "
    "No real patients or Trust data are displayed."
)
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
    "Fin Number",
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
        "Fin Number": "Fin Number",
        "Specialty": "Specialty",
        "Appointment Location Text": "Location",
        "Appointment Type": "Appointment Type",
        "DNA_Forecast_Text": "Expected DNA",
    }
)

# # Display filtered data
# st.dataframe(filtered_data_renamed,
#              use_container_width=True)


# Display filtered data
st.dataframe(
    filtered_data_renamed,
    column_config={
        "Appointment Date": st.column_config.DateColumn(
            "Appointment Date",
            format="DD/MM/YYYY",
        ),
        "MRN": st.column_config.TextColumn(),
        "Mobile Number": st.column_config.TextColumn(),
        #'Mobile Number (GP Record)': st.column_config.TextColumn(),
        "Fin Number": st.column_config.TextColumn(),
        "Specialty": st.column_config.TextColumn(),
        "Location": st.column_config.TextColumn(),
        "Appointment Type": st.column_config.TextColumn(),
        "Expected DNA": st.column_config.TextColumn(),
        "Confidence": st.column_config.TextColumn(),
        "Probabilty": st.column_config.NumberColumn(
            "DNA Risk", format="%.2f%%", help="Probability of DNA"
        ),
    },
)


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

    st.dataframe(styler, use_container_width=True)
else:
    st.info("No appointment data available for the selected filters.")
# --- END of finalized visualization ---
