"""Weekly summary view: appointments and forecast missed appointments by week and weekday.

Pure pandas, so the Streamlit tab only lays things out. Each measure is a 0/1 column of
the forecast data (high-risk flag, yes/no forecast) whose mean is a rate.
"""

from dataclasses import dataclass

import pandas as pd
from matplotlib.colors import to_rgb

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass(frozen=True)
class Measure:
    column: str  # 0/1 forecast column; its mean is the rate
    cell_label: str  # what the table cells count
    day_phrase: str  # drill-down heading wording
    rate_phrase: str  # summary sentence wording, with {rate}
    thresholds: tuple  # (green below, red above)
    help: str
    repeat_label: str  # metric card label for patients counted more than once


MEASURES = {
    "Very high risk (70% risk or more)": Measure(
        column="High_Risk_DNA",
        cell_label="at very high risk",
        day_phrase="at very high risk of being missed (70% risk or more)",
        rate_phrase="{rate} of appointments are at very high risk of being missed (70% or more)",
        thresholds=(0.02, 0.06),
        help="Only appointments with a predicted risk of not attending of 70% or more (the Very High risk band), "
        "the patients phoned in the real Physiotherapy pilot. A short, strict list: it catches the fewest "
        "missed appointments but wastes the fewest calls.",
        repeat_label="Patients flagged more than once",
    ),
    "Likely to be missed (20% risk or more)": Measure(
        column="DNA_Forecast",
        cell_label="at 20% risk or more",
        day_phrase="at 20% risk or more of being missed",
        rate_phrase="{rate} of appointments have a predicted risk of being missed of 20% or more",
        thresholds=(0.05, 0.15),
        help="Every appointment with a predicted risk of not attending of 20% or more (about two and a half "
        "times the average): the model's yes/no forecast, as in the 'Expected DNA' column in the "
        "appointment list. A longer list that catches more missed appointments but takes more calls, "
        "many to people who would have come anyway.",
        repeat_label="Patients forecast to miss more than one",
    ),
}


def _week_and_day(df):
    dates = pd.to_datetime(df["Activity_Date"])
    week_end = (
        dates.dt.to_period("W-SUN").dt.end_time.dt.normalize().rename("Week ending")
    )
    day = dates.dt.day_name().rename("Day")
    return week_end, day


def weekly_tables(df, column):
    """Appointments, measure totals and rates by week ending (earliest first) and weekday.

    Weekdays with no appointments at all (e.g. weekends) are left out. A rate is NaN
    where a week has no appointments on that day.
    """
    week_end, day = _week_and_day(df)
    days = [d for d in DAYS if d in set(day)]
    counts = pd.crosstab(week_end, day).reindex(columns=days, fill_value=0)
    sums = (
        df[column]
        .groupby([week_end, day])
        .sum()
        .unstack()
        .reindex(index=counts.index, columns=days)
        .fillna(0)
    )
    rates = sums / counts.where(counts > 0)
    order = counts.index.sort_values()
    return counts.loc[order], sums.loc[order], rates.loc[order]


def headline(df, column):
    """Overall rate, the highest/lowest weekday, the highest week and the first week."""
    week_end, day = _week_and_day(df)
    by_day = df[column].groupby(day).mean().reindex([d for d in DAYS if d in set(day)])
    by_week = df[column].groupby(week_end).mean()
    overall = df[column].mean()
    first_week = df[week_end == by_week.index.min()]
    return dict(
        n=len(df),
        n_weeks=len(by_week),
        n_days=len(by_day),
        total=df[column].sum(),
        overall=overall,
        top_day=by_day.idxmax(),
        top_day_rate=by_day.max(),
        low_day=by_day.idxmin(),
        low_day_rate=by_day.min(),
        top_week=by_week.idxmax(),
        top_week_rate=by_week.max(),
        first_week=by_week.index.min(),
        first_week_n=len(first_week),
        first_week_total=first_week[column].sum(),
        first_week_rate=first_week[column].mean(),
        weekly_average_total=df[column].sum() / len(by_week),
    )


def repeat_patients(df, m):
    """Patients with more than one flagged appointment."""
    counted = df[df[m.column] == 1]
    per_patient = counted.groupby("MRN").size()
    repeat = counted[counted["MRN"].isin(per_patient.index[per_patient > 1])]
    # One patient's appointments together, the patient with the soonest appointment first
    soonest = repeat.groupby("MRN")["Activity_Date"].transform("min")
    rows = (
        repeat.assign(_soonest=soonest)
        .sort_values(["_soonest", "MRN", "Activity_Date"])
        .drop(columns="_soonest")
    )
    return dict(
        patients=repeat["MRN"].nunique(),
        appointments=len(repeat),
        expected=repeat[m.column].sum(),
        rows=rows,
    )


def week_title(week_end, today=None):
    """'This week' or 'Next week' where that applies, else 'Week ending 6 Dec'."""
    today = pd.Timestamp.today().normalize() if today is None else today
    monday = week_end - pd.Timedelta(days=6)
    this_monday = today - pd.Timedelta(days=today.dayofweek)
    if monday == this_monday:
        return "This week"
    if monday == this_monday + pd.Timedelta(days=7):
        return "Next week"
    return f"Week ending {week_label(week_end)}"


def week_label(week_end, year=False):
    return f"{week_end.day} {week_end:%b}" + (f" {week_end:%Y}" if year else "")


def summary_cards(h, measure):
    """Plain-English summary as (label, text) pairs, one per card; comparisons that don't apply
    under the filters are left out."""
    m = MEASURES[measure]
    weeks = f"{h['n_weeks']} week{'s' if h['n_weeks'] != 1 else ''}"
    cards = [
        (f"Across {weeks}", m.rate_phrase.format(rate=f"**{h['overall']:.0%}**") + ".")
    ]

    if h["n_days"] > 1:
        top, low = h["top_day_rate"], h["low_day_rate"]
        s = f"**{h['top_day']}s** are highest at {top:.0%}"
        if low > 0 and top / low >= 1.5:
            s += f", about {top / low:.1f} times **{h['low_day']}s** ({low:.0%})"
        else:
            s += f", and **{h['low_day']}s** lowest at {low:.0%}"
        cards.append(("By weekday", s + "."))

    if h["n_weeks"] > 1:
        cards.append(
            (
                "By week",
                f"The week ending **{week_label(h['top_week'])}** is highest at {h['top_week_rate']:.0%}.",
            )
        )
    return cards


def cell_style(rate, low, high, alpha=1.0):
    """Green up to `low`, red from `high` up, amber between: three flat colours, so the only
    thing that varies within a colour is `alpha`, its opacity (1 is solid). Amber takes black
    text only while solid enough, as a faded one lets the dark page show through. Compared as
    the whole percentage the cell shows, so a cell reading 2% is green when `low` is 2%, and
    one reading 6% is red when `high` is 6%."""
    if pd.isna(rate):
        return ""
    shown = round(rate * 100)
    if shown <= round(low * 100):
        rgb, text = to_rgb("#198754"), "white"
    elif shown >= round(high * 100):
        rgb, text = to_rgb("#dc3545"), "white"
    else:
        rgb, text = to_rgb("#e0a526"), "black" if alpha >= 0.6 else "white"
    r, g, b = (round(c * 255) for c in rgb)
    return f"background-color: rgba({r}, {g}, {b}, {alpha:.2f}); color: {text};"


MIN_ALPHA = 0.15  # the faintest a cell with a rate goes, so even one with no calls is visible


def cell_styles(rates, sums, low, high):
    """cell_style for every cell of the weekly grid: colour from the rate, and opacity from the
    number of calls (`sums`), from MIN_ALPHA for none up to solid for the busiest cell."""
    busiest = sums.to_numpy().max() if sums.size else 0
    alpha = MIN_ALPHA + (1 - MIN_ALPHA) * (sums / busiest if busiest else sums * 0)
    return pd.DataFrame(
        [
            [cell_style(rates.iloc[i, j], low, high, alpha.iloc[i, j]) for j in range(rates.shape[1])]
            for i in range(rates.shape[0])
        ],
        index=rates.index,
        columns=rates.columns,
    )
