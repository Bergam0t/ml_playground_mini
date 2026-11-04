"""Synthetic stand-in for [dbo].[vw_Combined_Future_OP_Forecast].

Everything is randomly generated - no real patient or Trust data. Phone numbers use the
Ofcom reserved drama range (07700 900xxx).

Two parts:
- generate_appointments() makes appointments and a hidden "true" chance of not attending.
  Only the offline build (build/train_and_explain.py) calls it: once for made-up past
  appointments to train a model on, and once for the future appointments the app shows.
- load_forecast() is what the app calls. It reads the build's output (data/forecast.csv),
  which holds the model's predictions, and turns each row's day number into a date.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

SITES = {"Royal London": 0.10, "Newham": 0.20, "Whipps Cross": 0.05, "St Bart's": -0.25}
LOCATIONS = {
    "Royal London": ["RL Outpatients Zone A", "RL Outpatients Zone B", "RL Cardiac Centre"],
    "Newham": ["Newham Outpatients", "Newham Diagnostic Hub"],
    "Whipps Cross": ["WX Outpatients Zone 1", "WX Outpatients Zone 2", "WX Eye Unit"],
    "St Bart's": ["Bart's Heart Centre", "Bart's Outpatients"],
}
SPECIALTIES = {
    "Cardiology": -0.25, "Dermatology": 0.10, "Diabetes & Endocrine": 0.20,
    "ENT": 0.05, "Gastroenterology": -0.05, "General Surgery": -0.15,
    "Gynaecology": 0.00, "Ophthalmology": 0.25, "Orthopaedics": -0.05,
    "Respiratory": -0.10, "Rheumatology": -0.20, "Urology": -0.10,
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_EFFECT = dict(zip(DAYS, [0.60, 0.05, -0.40, -0.15, 0.45]))
APPOINTMENT_TYPES = {"New": 0.3, "Follow Up": -0.1}

# Model inputs: column -> label shown on the waterfall chart
FEATURES = {
    "Previous_DNAs": "Missed appointments (last 12 months)",
    "Lead_Time_Days": "Days since booking",
    "Age": "Age",
    "Deprivation_Decile": "Deprivation decile (1 = most deprived)",
    "Appointment Type": "Appointment type",
    "Specialty": "Specialty",
    "Reporting_Site": "Site",
    "Day_Of_Week": "Day of the week",
}
# Categorical model inputs and their categories (fixed, so one-hot columns always line up)
CATEGORIES = {
    "Appointment Type": list(APPOINTMENT_TYPES),
    "Specialty": list(SPECIALTIES),
    "Reporting_Site": list(SITES),
    "Day_Of_Week": DAYS,
}


def generate_appointments(n=3200, seed=42, weeks=6):
    """Appointments with features, IDs and a hidden true DNA probability (`p_true`).

    Each row has a `day_index` (0 = the first Monday) rather than a date, so the
    weekday of a row never changes when the dates roll forward.
    """
    rng = np.random.default_rng(seed)
    di = rng.integers(0, weeks * 5, n)
    df = pd.DataFrame({
        "day_index": di,
        "Day_Of_Week": np.array(DAYS)[di % 5],
        "Reporting_Site": rng.choice(list(SITES), n, p=[0.35, 0.25, 0.20, 0.20]),
        "Specialty": rng.choice(list(SPECIALTIES), n),
        "Appointment Type": rng.choice(list(APPOINTMENT_TYPES), n, p=[0.35, 0.65]),
    })
    df["Appointment Location Text"] = [rng.choice(LOCATIONS[s]) for s in df["Reporting_Site"]]

    # Patient history and circumstances. A minority of patients miss appointments
    # repeatedly, so previous DNAs come from a mixture.
    # Some weeks (e.g. school holidays) have more patients who often miss appointments,
    # which makes the weekly DNA rates vary.
    week_mix = np.exp(rng.normal(0, 0.45, weeks))[di // 5]
    habitual = rng.random(n) < np.clip(0.16 * week_mix, 0, 0.6)
    df["Previous_DNAs"] = np.minimum(rng.poisson(np.where(habitual, 2.2, 0.25)), 8)
    df["Lead_Time_Days"] = np.clip(np.round(rng.lognormal(np.log(35), 0.75, n)), 2, 200).astype(int)
    df["Age"] = np.clip(np.round(rng.normal(52, 18, n)), 18, 95).astype(int)
    df["Deprivation_Decile"] = rng.choice(np.arange(1, 11), n, p=np.linspace(0.14, 0.06, 10))

    logit = 1.5 * (
        df["Reporting_Site"].map(SITES).to_numpy()
        + df["Specialty"].map(SPECIALTIES).to_numpy()
        + df["Appointment Type"].map(APPOINTMENT_TYPES).to_numpy()
        + df["Day_Of_Week"].map(DAY_EFFECT).to_numpy()
        + 0.65 * np.minimum(df["Previous_DNAs"].to_numpy(), 5)
        + 0.55 * (np.log(df["Lead_Time_Days"].to_numpy()) - np.log(35))
        - 0.025 * (df["Age"].to_numpy() - 52)
        - 0.10 * (df["Deprivation_Decile"].to_numpy() - 5.5)
    ) + rng.normal(0, 0.5, n)
    # calibrate intercept so ~26% of appointments have a true DNA chance of 50% or more
    # (the model, which cannot see the noise, then forecasts about 22% as DNAs)
    lo, hi = -8.0, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        share = ((1 / (1 + np.exp(-(logit + mid)))) >= 0.5).mean()
        lo, hi = (mid, hi) if share < 0.26 else (lo, mid)
    df["p_true"] = 1 / (1 + np.exp(-(logit + mid)))

    df["MRN"] = ["MRN" + str(x).zfill(6) for x in rng.permutation(n) + 100000]
    df["Mobile Telephone"] = ["07700900" + str(x).zfill(3) for x in rng.integers(0, 1000, n)]
    df["Fin Number"] = ["99" + str(x).zfill(8) for x in rng.integers(0, 10**8, n)]
    df.insert(0, "row_id", np.arange(n))
    return df


def first_monday(today=None):
    """The next Monday after today (a week on, if today is a Monday)."""
    today = pd.Timestamp.today().normalize() if today is None else today
    return today + pd.Timedelta(days=7 - today.dayofweek)


def load_forecast():
    """The future appointments with the model's predictions, dated from next Monday."""
    df = pd.read_csv(DATA_DIR / "forecast.csv", dtype={"Mobile Telephone": str, "Fin Number": str})
    days = pd.bdate_range(first_monday(), periods=int(df["day_index"].max()) + 1)
    df["Activity_Date"] = days[df["day_index"].to_numpy()]
    return df.sort_values(["Activity_Date", "Reporting_Site"]).reset_index(drop=True)
