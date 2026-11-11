"""Per-appointment explanations from the exported SHAP values (pandas only; no Streamlit).

Ported from lambda_ml_streamlit_exercise/app/explain.py.
"""
import pandas as pd

from demo_data import FEATURES


def display_value(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def patient_explanation(row: pd.Series, shap_row: pd.Series, base_pp: float, top: int = 6) -> dict:
    """Contributions in percentage points for one appointment. base + sum(contributions) = f(x).

    `row` is the appointment's row from the forecast, `shap_row` its row from shap_values.csv.
    The `top` biggest contributions are shown; the rest are rolled into one "other factors" bar.
    """
    items = [
        {
            "feature": k,
            "label": f"{FEATURES[k]} = {display_value(row['Appointment_Time'] if k == 'Time_Slot' else row[k])}",
            "value": float(shap_row[k]),
        }
        for k in FEATURES
    ]
    items.sort(key=lambda i: abs(i["value"]), reverse=True)
    shown, rest = items[:top], items[top:]
    other = (
        {"label": f"{len(rest)} other factor{'s' if len(rest) > 1 else ''}", "value": float(sum(i["value"] for i in rest))}
        if rest
        else None
    )
    return {
        "base": base_pp,
        "fx": float(row["DNA_Prob_Forecast"]) * 100,
        "shown": shown,
        "other": other,
        "all": items,
    }


def group_importance(shap_rows: pd.DataFrame) -> list:
    """Each factor's mean |SHAP| in percentage points across many appointments, biggest first.

    How far the factor moves an appointment's predicted risk on average, in either direction.
    (The signed mean would let pushes up and down cancel out.)
    """
    items = [{"label": label, "value": float(shap_rows[k].abs().mean())} for k, label in FEATURES.items()]
    items.sort(key=lambda i: i["value"], reverse=True)
    return items
