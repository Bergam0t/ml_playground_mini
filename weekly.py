"""Weekly summary: appointments and forecast missed appointments by week and weekday.

Pure pandas, so the Streamlit tab only lays things out. Each measure is a column of
the forecast data whose mean is a rate: a 0/1 flag (high-risk flag, yes/no forecast)
or the predicted risk itself, whose sum is the expected number of missed appointments.
"""
from dataclasses import dataclass

import matplotlib
import pandas as pd
from matplotlib.colors import to_hex

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]



@dataclass(frozen=True)
class Measure:
    column: str  # forecast column; its mean is the rate
    count_label: str  # metric card label for its total
    cell_label: str  # what the table cells count
    rate_phrase: str  # summary sentence wording, with {rate}
    thresholds: tuple  # (green below, red above)
    flag: bool  # a 0/1 flag (else a probability)
    help: str
    repeat_label: str  # metric card label for patients counted more than once
    day_phrase: str = ""  # drill-down heading wording (flags only)


MEASURES = {
    "High-risk flagged": Measure(
        column="High_Risk_DNA",
        count_label="High-risk flagged",
        cell_label="flagged",
        day_phrase="high-risk flagged",
        rate_phrase="{rate} of appointments are high-risk flagged",
        thresholds=(0.08, 0.20),
        flag=True,
        help="Predicted risk over 70% and confidence over 70%.",
        repeat_label="Patients flagged more than once",
    ),
    "Forecast DNAs": Measure(
        column="DNA_Forecast",
        count_label="Forecast DNAs",
        cell_label="forecast DNAs",
        day_phrase="forecast not to attend",
        rate_phrase="the model forecasts that {rate} of appointments will be missed",
        thresholds=(0.15, 0.35),
        flag=True,
        help="The model's yes/no forecast, as in the Appointments tab's 'Expected DNA' column.",
        repeat_label="Patients forecast to miss more than one",
    ),
    "Average predicted risk": Measure(
        column="DNA_Prob_Forecast",
        count_label="Expected missed appointments",
        cell_label="expected DNAs",
        rate_phrase="the average predicted risk of not attending is {rate}",
        thresholds=(0.25, 0.40),
        flag=False,
        help="The average of the predicted risks. Adding the risks up gives the expected "
        "number of missed appointments.",
        repeat_label="Patients with more than one appointment",
    ),
}


def _week_and_day(df):
    dates = pd.to_datetime(df["Activity_Date"])
    week_end = dates.dt.to_period("W-SUN").dt.end_time.dt.normalize().rename("Week ending")
    day = dates.dt.day_name().rename("Day")
    return week_end, day


def weekly_tables(df, column):
    """Appointments, measure totals and rates by week ending (newest first) and weekday.

    Weekdays with no appointments at all (e.g. weekends) are left out. A rate is NaN
    where a week has no appointments on that day.
    """
    week_end, day = _week_and_day(df)
    days = [d for d in DAYS if d in set(day)]
    counts = pd.crosstab(week_end, day).reindex(columns=days, fill_value=0)
    sums = (
        df[column].groupby([week_end, day]).sum().unstack()
        .reindex(index=counts.index, columns=days).fillna(0)
    )
    rates = sums / counts.where(counts > 0)
    order = counts.index.sort_values(ascending=False)
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
    """Patients with more than one counted appointment: more than one flagged one for a
    0/1 measure, or more than one appointment of any kind for the predicted risk."""
    counted = df[df[m.column] == 1] if m.flag else df
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


def pts_vs_average(rate, overall):
    return f"{(rate - overall) * 100:+.1f} pts vs average"


def week_label(week_end, year=False):
    return f"{week_end.day} {week_end:%b}" + (f" {week_end:%Y}" if year else "")


def summary_sentence(h, measure):
    """Plain-English summary; comparisons that don't apply under the filters are left out."""
    m = MEASURES[measure]
    weeks = f"{h['n_weeks']} week{'s' if h['n_weeks'] != 1 else ''}"
    parts = [f"Across {weeks}, " + m.rate_phrase.format(rate=f"**{h['overall']:.0%}**") + "."]

    if h["n_days"] > 1:
        top, low = h["top_day_rate"], h["low_day_rate"]
        s = f"**{h['top_day']}s** are highest at {top:.0%}"
        if low > 0 and top / low >= 1.5:
            s += f", about {top / low:.1f} times **{h['low_day']}s** ({low:.0%})"
        else:
            s += f", and **{h['low_day']}s** lowest at {low:.0%}"
        parts.append(s + ".")

    if h["n_weeks"] > 1:
        parts.append(
            f"The week ending **{week_label(h['top_week'])}** is highest at {h['top_week_rate']:.0%}."
        )
    return " ".join(parts)


def cell_style(rate, low, high):
    """Green below `low`, red above `high`, a red-yellow-green gradient between."""
    if pd.isna(rate):
        return ""
    if rate < low:
        return "background-color: #198754; color: white;"
    if rate > high:
        return "background-color: #dc3545; color: white;"
    cmap = matplotlib.colormaps["RdYlGn_r"]
    norm = matplotlib.colors.Normalize(vmin=low, vmax=high)
    return f"background-color: {to_hex(cmap(norm(rate)))}; color: black;"
