"""Synthetic stand-in for [dbo].[vw_Combined_Future_OP_Forecast].

Produces the same columns OP_Forecast_App.py reads. Everything is randomly
generated - no real patient or Trust data. Phone numbers use the Ofcom
reserved drama range (07700 900xxx).
"""
import numpy as np
import pandas as pd

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
DAY_EFFECT = np.array([0.85, 0.10, -0.55, -0.20, 0.65])  # Mon..Fri


def generate_forecast(n=3200, seed=42, weeks=6):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    days = pd.bdate_range(start, periods=weeks * 5)
    week_effect = rng.normal(0, 0.30, weeks)

    di = rng.integers(0, len(days), n)
    df = pd.DataFrame({
        "Activity_Date": days[di],
        "Reporting_Site": rng.choice(list(SITES), n, p=[0.35, 0.25, 0.20, 0.20]),
        "Specialty": rng.choice(list(SPECIALTIES), n),
        "Appointment Type": rng.choice(["New", "Follow Up"], n, p=[0.35, 0.65]),
    })
    df["Appointment Location Text"] = [rng.choice(LOCATIONS[s]) for s in df["Reporting_Site"]]

    logit = (
        df["Reporting_Site"].map(SITES).to_numpy()
        + df["Specialty"].map(SPECIALTIES).to_numpy()
        + np.where(df["Appointment Type"] == "New", 0.3, -0.1)
        + DAY_EFFECT[days[di].dayofweek.to_numpy()]
        + week_effect[di // 5]
        + rng.normal(0, 1.5, n)
    )
    # calibrate intercept so ~22% of appointments are forecast as DNA
    lo, hi = -8.0, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        share = ((1 / (1 + np.exp(-(logit + mid)))) >= 0.5).mean()
        lo, hi = (mid, hi) if share < 0.22 else (lo, mid)
    p = 1 / (1 + np.exp(-(logit + mid)))

    df["DNA_Prob_Forecast"] = p.round(3)
    df["Confidence_Score"] = np.clip(np.maximum(p, 1 - p) + rng.normal(0, 0.05, n), 0.5, 0.99).round(3)
    df["DNA_Forecast"] = (p >= 0.5).astype(int)
    df["MRN"] = ["MRN" + str(x).zfill(6) for x in rng.permutation(n) + 100000]
    df["Mobile Telephone"] = ["07700900" + str(x).zfill(3) for x in rng.integers(0, 1000, n)]
    df["Fin Number"] = ["99" + str(x).zfill(8) for x in rng.integers(0, 10**8, n)]
    return df.sort_values(["Activity_Date", "Reporting_Site"]).reset_index(drop=True)
