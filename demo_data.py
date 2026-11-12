"""Synthetic stand-in for [dbo].[vw_Combined_Future_OP_Forecast].

Everything is randomly generated - no real patient or Trust data. The trust and its sites
are fictional (TRUST_NAME, SITES). Mobile numbers are random UK mobile numbers, so some may
happen to match real ones.

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

# A fictional trust and its sites, so the made-up differences between sites can't be read as
# findings about real hospitals. They replaced real hospital names one for one: each site keeps
# its place in these lists and its number of locations (the random draws pick by position), and
# the sites sort in the same alphabetical order, so every number, and the order of rows, stayed
# exactly as it was. Keep both true if you rename them again.
TRUST_NAME = "Riverbourne Hospitals"
SITES = {"Brenmoor": 0.10, "Ashby Vale": 0.20, "Dunmere": 0.05, "Calderwood": -0.25}
LOCATIONS = {
    "Brenmoor": ["BM Outpatients Zone A", "BM Outpatients Zone B", "BM Cardiac Centre"],
    "Ashby Vale": ["Ashby Vale Outpatients", "Ashby Vale Diagnostic Hub"],
    "Dunmere": ["DM Outpatients Zone 1", "DM Outpatients Zone 2", "DM Eye Unit"],
    "Calderwood": ["Calderwood Heart Centre", "Calderwood Outpatients"],
}
SPECIALTIES = {
    "Cardiology": -0.25, "Dermatology": 0.10, "Diabetes & Endocrine": 0.20,
    "ENT": 0.05, "Gastroenterology": -0.05, "General Surgery": -0.15,
    "Gynaecology": 0.00, "Ophthalmology": 0.25, "Orthopaedics": -0.05,
    "Physiotherapy": 0.20, "Respiratory": -0.10, "Rheumatology": -0.20, "Urology": -0.10,
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DAY_EFFECT = dict(zip(DAYS, [0.60, 0.05, -0.40, -0.15, 0.45]))
APPOINTMENT_TYPES = {"New": 0.3, "Follow Up": -0.1}
# Clinic starts in hours: the model sees the hour an appointment starts in. Mid-morning (10-12)
# and late afternoon (3-5) are worst; early morning and early afternoon are best.
TIME_EFFECT = {
    "08:00": -0.25, "09:00": -0.15, "10:00": 0.30, "11:00": 0.30, "12:00": -0.10,
    "13:00": -0.20, "14:00": -0.10, "15:00": 0.30, "16:00": 0.30,
}
# Extra risk for Monday before midday, on top of Monday and the time of day
MONDAY_MORNING_EFFECT = 0.50
# Appointment starts: every 15 minutes from 08:30 to 16:45, fewer over lunch
SLOT_TIMES = [f"{m // 60:02d}:{m % 60:02d}" for m in range(8 * 60 + 30, 16 * 60 + 46, 15)]
SLOT_WEIGHTS = np.array([0.5 if "12:30" <= t < "13:30" else 1.0 for t in SLOT_TIMES])

FIRST_NAMES = [
    "Amelia", "Oliver", "Fatima", "Mohammed", "Priya", "Jack", "Aisha", "George", "Grace", "Tariq",
    "Sophie", "Ahmed", "Emily", "Harry", "Zainab", "Daniel", "Ruth", "Samir", "Chloe", "Joseph",
    "Hannah", "Ibrahim", "Lucy", "Arjun", "Margaret", "Kwame", "Sarah", "Raj", "Beatrice", "Tomasz",
    "Naomi", "Hassan", "Ellie", "Marcus", "Leila", "Peter", "Anita", "Thomas", "Maryam", "William",
]
LAST_NAMES = [
    "Smith", "Khan", "Patel", "Jones", "Ali", "Taylor", "Begum", "Brown", "Hussain", "Williams",
    "Ahmed", "Evans", "Singh", "Wilson", "Rahman", "Davies", "Okafor", "Thomas", "Islam", "Roberts",
    "Nowak", "Johnson", "Chowdhury", "Walker", "Mensah", "Wright", "Sharma", "Hall", "Costa", "Green",
]

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
    "Time_Slot": "Appointment time",
    "Journey_Time_Mins": "Predicted journey time (minutes)",
    "Weeks_Since_Last": "Weeks since last seen (or referred)",
    "Treatment_Ended": "Treatment already finished",
}
# Categorical model inputs and their categories (fixed, so one-hot columns always line up)
CATEGORIES = {
    "Appointment Type": list(APPOINTMENT_TYPES),
    "Specialty": list(SPECIALTIES),
    "Reporting_Site": list(SITES),
    "Day_Of_Week": DAYS,
    "Time_Slot": list(TIME_EFFECT),
    "Treatment_Ended": ["No", "Yes"],
}


SITE_P = [0.35, 0.25, 0.20, 0.20]
# Share of appointments missed. Recent NHS outpatient figures are about 6.4%; the demo
# Trust is a little worse than average.
DNA_RATE = 0.08
# How many appointments a patient has in the period, and how likely each count is
VISITS_PER_PATIENT, VISITS_P = [1, 2, 3], [0.75, 0.19, 0.06]
# Patients who often miss: about this share of them, with this many missed appointments in
# the last 12 months on average (others average 0.25)
HABITUAL_SHARE = 0.08
HABITUAL_MEAN = 5.0
# The effects below are set so the model reflects the real model's common reasons (long or
# difficult journeys, long waits between appointments, appointments after treatment had
# finished) more than missed appointments before, while, as in the real Physiotherapy pilot
# (pilot.py), about 84% of the appointments it flags as high risk (70% or more) are missed
# without a call, and its test AUC stays near the "around 86%" the real model is quoted at.
# The build checks these. Found by searching on the build's fixed seeds: only about 60 past
# test appointments are flagged, so small changes here move those checks by a lot (about 10
# points of the 84% and 0.01 of the AUC) as the model refits. Rerun the build after any change.

# Missed appointments before: adds this much to the log-odds (times 1.5, below) for each one.
# Still a strong clue, but no longer the one that outweighs everything else.
PREVIOUS_DNA_EFFECT = 0.417
# Predicted journey time to the appointment: one of the common reasons the real model gave
# ("long or difficult journeys"). Adds this much to the log-odds (times 1.5) for each step
# of the log of the journey time away from a typical 25 minutes, so it counts more and more
# slowly as journeys get longer.
JOURNEY_EFFECT = 1.214
TYPICAL_JOURNEY_MINS = 25
# Weeks since the patient was last seen in this specialty (a follow-up), or since they were
# referred (a new appointment): "long waits between appointments", one of the real model's
# common reasons. Adds this much to the log-odds (times 1.5) for each step of the log of the
# gap away from a typical gap.
GAP_EFFECT = 0.947
TYPICAL_GAP_WEEKS = 10
# A follow-up booked after the course of treatment had finished ("appointments after treatment
# had finished", another of the real model's common reasons): this share of follow-ups, adding
# this much to the log-odds (times 1.5). Rare, but a strong sign: about 4 in 10 are missed.
TREATMENT_ENDED_SHARE = 0.033
TREATMENT_ENDED_EFFECT = 2.475
# Patients who've "moved on" (feel better, or have gone elsewhere): about this share of them.
# Their follow-ups are much more often after treatment had finished (this share), and their
# gaps longer (this many times), so these signals pile up in the same few patients. That gives
# some of the very high risks a call list needs, as habitual missers do.
MOVED_ON_SHARE = 0.04
MOVED_ON_ENDED = 0.866
MOVED_ON_GAP = 1.854
# What the clues can't see: the spread of each appointment's chance around them
NOISE_SD = 1.432


def generate_appointments(n=3200, seed=42, weeks=6):
    """Appointments with features, IDs and a hidden true DNA probability (`p_true`).

    Most patients have one appointment in the period; some have two or three, on
    different days. A patient keeps their ID, mobile number, age, deprivation decile
    and missed-appointment history across all of them, and usually their site and
    specialty too (a repeat in the same specialty is a follow-up).

    Patient names and appointment times come from a separate random stream, so adding them
    left every other column as it was. A patient keeps their name across appointments.
    Journey times have a stream of their own too, so adding them changed only the risk, and
    so do the gap since last seen and whether treatment had already finished.

    Each row has a `day_index` (0 = the first Monday) rather than a date, so the
    weekday of a row never changes when the dates roll forward.
    """
    rng = np.random.default_rng(seed)
    n_days = weeks * 5

    # ---- Patients: draw more than enough, keep just enough for n appointments, and
    # trim the last patient's count so the total is exactly n
    visits = rng.choice(VISITS_PER_PATIENT, n, p=VISITS_P)
    visits = visits[: np.searchsorted(np.cumsum(visits), n) + 1]
    visits[-1] -= visits.sum() - n
    m = len(visits)

    # Random UK mobile numbers (071-075, 077-079; 070 and 076 are not mobile ranges).
    # Made up, but some may match real numbers.
    phones = [f"07{t}{r:08d}" for t, r in
              zip(rng.choice(list("12345789"), m), rng.integers(0, 10**8, m))]
    home_site = rng.choice(list(SITES), m, p=SITE_P)
    first_specialty = rng.choice(list(SPECIALTIES), m)
    age = np.clip(np.round(rng.normal(52, 18, m)), 18, 95).astype(int)
    decile = rng.choice(np.arange(1, 11), m, p=np.linspace(0.14, 0.06, 10))

    # ---- Appointments: each patient's are on different days, earliest first
    patient = np.repeat(np.arange(m), visits)
    visit_no = np.concatenate([np.arange(k) for k in visits])
    di = np.concatenate([np.sort(rng.choice(n_days, k, replace=False)) for k in visits])

    same_site = rng.random(n) < 0.9
    site = np.where(same_site, home_site[patient], rng.choice(list(SITES), n, p=SITE_P))
    same_specialty = (visit_no == 0) | (rng.random(n) < 0.75)
    specialty = np.where(same_specialty, first_specialty[patient], rng.choice(list(SPECIALTIES), n))
    appt_type = rng.choice(list(APPOINTMENT_TYPES), n, p=[0.35, 0.65])
    appt_type[(visit_no > 0) & same_specialty] = "Follow Up"

    df = pd.DataFrame({
        "day_index": di,
        "Day_Of_Week": np.array(DAYS)[di % 5],
        "Reporting_Site": site,
        "Specialty": specialty,
        "Appointment Type": appt_type,
    })
    # Names and clinic times: own random stream (see above)
    rng_extra = np.random.default_rng([seed, 1])
    names = [f"{f} {l}" for f, l in zip(rng_extra.choice(FIRST_NAMES, m), rng_extra.choice(LAST_NAMES, m))]
    df["Patient Name"] = np.array(names)[patient]
    df["Appointment_Time"] = rng_extra.choice(SLOT_TIMES, n, p=SLOT_WEIGHTS / SLOT_WEIGHTS.sum())
    df["Time_Slot"] = df["Appointment_Time"].str[:2] + ":00"
    df["Appointment Location Text"] = [rng.choice(LOCATIONS[s]) for s in df["Reporting_Site"]]

    # Patient history and circumstances. A minority of patients miss appointments
    # repeatedly, so previous DNAs come from a mixture.
    # Some weeks (e.g. school holidays) have more patients who often miss appointments,
    # which makes the weekly DNA rates vary. A patient counts in the week of their
    # first appointment.
    week_mix = np.exp(rng.normal(0, 0.45, weeks))[di[visit_no == 0] // 5]
    habitual = rng.random(m) < np.clip(HABITUAL_SHARE * week_mix, 0, 0.6)
    df["Previous_DNAs"] = np.minimum(rng.poisson(np.where(habitual, HABITUAL_MEAN, 0.25)), 8)[patient]
    df["Lead_Time_Days"] = np.clip(np.round(rng.lognormal(np.log(35), 0.75, n)), 2, 200).astype(int)
    df["Age"] = age[patient]
    df["Deprivation_Decile"] = decile[patient]

    # Predicted journey time: own random stream (see above). Each patient has a usual journey
    # to their usual site, a little longer from more deprived areas (more often by bus), and
    # an appointment at another site takes 20 to 40 minutes more.
    rng_journey = np.random.default_rng([seed, 2])
    usual = rng_journey.lognormal(np.log(TYPICAL_JOURNEY_MINS), 0.45, m) * np.exp(0.04 * (5.5 - decile))
    extra = np.where(same_site, 0, rng_journey.uniform(20, 40, n))
    df["Journey_Time_Mins"] = np.clip(np.round(usual[patient] + extra), 5, 120).astype(int)

    # Gap since last seen (or referred), and whether treatment had already finished: own random
    # stream (see above), so adding them changed only the risk. Follow-ups spread more widely
    # than referral waits. The gap is at least the time since booking: it was booked after.
    rng_course = np.random.default_rng([seed, 3])
    moved_on = (rng_course.random(m) < MOVED_ON_SHARE)[patient]
    follow_up = (df["Appointment Type"] == "Follow Up").to_numpy()
    gap = np.where(
        follow_up,
        rng_course.lognormal(np.log(TYPICAL_GAP_WEEKS), 0.75, n),
        rng_course.lognormal(np.log(8), 0.5, n),
    ) * np.where(moved_on, MOVED_ON_GAP, 1.0)
    gap = np.maximum(gap, np.ceil(df["Lead_Time_Days"].to_numpy() / 7))
    df["Weeks_Since_Last"] = np.clip(np.round(gap), 1, 104).astype(int)
    ended = follow_up & (rng_course.random(n) < np.where(moved_on, MOVED_ON_ENDED, TREATMENT_ENDED_SHARE))
    df["Treatment_Ended"] = np.where(ended, "Yes", "No")

    logit = 1.5 * (
        df["Reporting_Site"].map(SITES).to_numpy()
        + df["Specialty"].map(SPECIALTIES).to_numpy()
        + df["Appointment Type"].map(APPOINTMENT_TYPES).to_numpy()
        + df["Day_Of_Week"].map(DAY_EFFECT).to_numpy()
        + df["Time_Slot"].map(TIME_EFFECT).to_numpy()
        + MONDAY_MORNING_EFFECT * ((df["Day_Of_Week"] == "Monday") & (df["Time_Slot"] < "12:00")).to_numpy()
        + PREVIOUS_DNA_EFFECT * df["Previous_DNAs"].to_numpy()
        + 0.55 * (np.log(df["Lead_Time_Days"].to_numpy()) - np.log(35))
        - 0.025 * (df["Age"].to_numpy() - 52)
        - 0.10 * (df["Deprivation_Decile"].to_numpy() - 5.5)
        + JOURNEY_EFFECT * (np.log(df["Journey_Time_Mins"].to_numpy()) - np.log(TYPICAL_JOURNEY_MINS))
        + GAP_EFFECT * (np.log(df["Weeks_Since_Last"].to_numpy()) - np.log(TYPICAL_GAP_WEEKS))
        + TREATMENT_ENDED_EFFECT * ended
    ) + rng.normal(0, NOISE_SD, n)
    # calibrate the intercept so the average true DNA chance is DNA_RATE
    lo, hi = -12.0, 4.0
    for _ in range(50):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit + mid)))).mean()
        lo, hi = (mid, hi) if rate < DNA_RATE else (lo, mid)
    df["p_true"] = 1 / (1 + np.exp(-(logit + mid)))

    mrns = np.array(["P" + str(x).zfill(6) for x in rng.permutation(m) + 100000])
    df["MRN"] = mrns[patient]
    df["Mobile Telephone"] = np.array(phones)[patient]
    df["Fin Number"] = ["99" + str(x).zfill(8) for x in rng.integers(0, 10**8, n)]  # one per appointment

    # shuffle so a patient's appointments are not next to each other
    df = df.iloc[rng.permutation(n)].reset_index(drop=True)
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
