"""Costs: what reminder calls cost in staff time, and what a missed appointment is worth.

Pure Python/pandas, so the Streamlit app only lays things out. Every figure here is an
illustrative default that the app lets users change.

Staff time: a band 5 admin making the calls. Agenda for Change 2025/26 pay, middle pay
point of band 5, plus the employer's National Insurance (15% above the £5,000 secondary
threshold, from April 2025) and the employer pension contribution trusts pay (14.38%; the
rest of the 23.7% is paid centrally). Productive hours are 37.5 a week over 52.14 weeks,
less 27 days' annual leave, 8 bank holidays and about 4% for sickness and training. Check
the salary against the latest pay award.

Missed appointments: illustrative values for an unused outpatient slot, by specialty and
first (New) or follow-up attendance, loosely in line with NHS Payment Scheme
consultant-led outpatient prices. A price is not a cost, and a Trust should replace these
with its own NHS Payment Scheme or patient-level costing (PLICS) figures.
"""
import pandas as pd

# Agenda for Change band 5, 2025/26, with employer on-costs and productive hours
BAND5 = dict(
    salary=33_487,  # middle pay point (2 to 4 years)
    ni_rate=0.15,
    ni_threshold=5_000,
    pension_rate=0.1438,
    hours_per_week=37.5,
    weeks_per_year=52.14,
    leave_days=27 + 8,  # annual leave and bank holidays
    other_absence=0.04,  # sickness and training
)

MINUTES_PER_CALL = 3  # average, including unanswered attempts and noting the outcome

APPOINTMENT_TYPES = ["New", "Follow Up"]

# £ per missed appointment: (New, Follow Up)
DNA_COSTS = {
    "Cardiology": (240, 120),
    "Dermatology": (170, 100),
    "Diabetes & Endocrine": (240, 120),
    "ENT": (170, 100),
    "Gastroenterology": (250, 130),
    "General Surgery": (190, 120),
    "Gynaecology": (210, 140),
    "Ophthalmology": (170, 100),
    "Orthopaedics": (170, 110),
    "Respiratory": (250, 130),
    "Rheumatology": (260, 120),
    "Urology": (190, 100),
}


def admin_hourly_cost(salary, ni_rate, ni_threshold, pension_rate, hours_per_week,
                      weeks_per_year, leave_days, other_absence):
    """Employer cost per productive hour, with the parts that make it up."""
    ni = max(salary - ni_threshold, 0) * ni_rate
    pension = salary * pension_rate
    total = salary + ni + pension
    paid_hours = hours_per_week * weeks_per_year
    leave_hours = leave_days * hours_per_week / 5
    hours = (paid_hours - leave_hours) * (1 - other_absence)
    return dict(salary=salary, ni=ni, pension=pension, total=total, hours=hours,
                hourly=total / hours)


def dna_cost_table():
    """The default costs as a table: Specialty, New, Follow Up."""
    return pd.DataFrame(
        [(s, new, fu) for s, (new, fu) in DNA_COSTS.items()],
        columns=["Specialty", *APPOINTMENT_TYPES],
    )


def appointment_costs(df, table, share_lost=1.0):
    """£ a missed appointment would cost, for each appointment in `df`.

    `table` is a dna_cost_table() (possibly edited). A specialty missing from it, or a
    blank cost, counts as £0.
    """
    long = (
        table.set_index("Specialty")[APPOINTMENT_TYPES]
        .stack()
        .rename("cost")
    )
    keys = pd.MultiIndex.from_arrays([df["Specialty"], df["Appointment Type"]])
    cost = long.reindex(keys).fillna(0).to_numpy(dtype=float)
    return pd.Series(cost * share_lost, index=df.index)


def expected_dna_cost(df, cost_column="DNA_Cost"):
    """Expected cost of the missed appointments: each predicted risk times its cost."""
    return float((df["DNA_Prob_Forecast"] * df[cost_column]).sum())


def cost_per_dna_reached(df, cost_column="DNA_Cost"):
    """Average cost of a missed appointment among `df`, weighted by predicted risk, so the
    patients more likely to miss count more. None if there is no risk to weight by."""
    p = df["DNA_Prob_Forecast"]
    return float((p * df[cost_column]).sum() / p.sum()) if p.sum() > 0 else None


def call_economics(calls, reached, effect, cost_per_call, minutes_per_call, dna_cost):
    """Staff time and cost of `calls`, and the value of the missed appointments they reach
    and might avoid.

    `reached` is how many calls reach someone who would otherwise miss; `effect` is the
    assumed share of them a call gets to attend (0 to 1); `dna_cost` is the average cost
    of a missed appointment among those reached. The break-even effect is the share the
    calls would need to get to attend to cover their own staff cost (None if they reach
    nobody, or over 1 if no effect could).
    """
    call_cost = calls * cost_per_call
    value_reached = reached * (dna_cost or 0)
    value_avoided = value_reached * effect
    return dict(
        staff_hours=calls * minutes_per_call / 60,
        call_cost=call_cost,
        value_reached=value_reached,
        value_avoided=value_avoided,
        net=value_avoided - call_cost,
        break_even_effect=call_cost / value_reached if value_reached > 0 else None,
    )
