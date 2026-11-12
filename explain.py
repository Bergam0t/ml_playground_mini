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
    The `top` biggest contributions are shown; the rest are rolled into one "other factors" bar,
    and listed under "rest" (the waterfall's hover on that bar names them).
    """
    items = [
        {
            "feature": k,
            "label": f"{FEATURES[k]} = {display_value(row['Appointment_Time'] if k == 'Time_Slot' else row[k])}",
            "value": float(shap_row[k]),
        }
        for k in FEATURES
    ]
    # Switched off for now: each factor compared with every upcoming appointment, for a richer
    # waterfall hover (see charts._waterfall_hover). Left out because, at the stand, a third kind
    # of percentage beside the risk and the bars was more likely to confuse than help. To bring it
    # back, add `shap_all: pd.DataFrame | None = None` to the arguments, pass
    # `shap_all=load_shap()` at both calls in streamlit_app.py, and uncomment this and
    # "n_compared" below. "share" is the fraction of appointments the factor pushes less far in
    # the same direction; "range" is its (lowest, highest) contribution across them all.
    # if shap_all is not None:
    #     for i in items:
    #         col = shap_all[i["feature"]]
    #         further = col < i["value"] if i["value"] >= 0 else col > i["value"]
    #         i["share"] = float(further.mean())
    #         i["range"] = (float(col.min()), float(col.max()))
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
        "rest": rest,
        "all": items,
        # "n_compared": None if shap_all is None else len(shap_all),
    }


def group_importance(shap_rows: pd.DataFrame) -> list:
    """Each factor's mean |SHAP| in percentage points across many appointments, biggest first.

    How far the factor moves an appointment's predicted risk on average, in either direction.
    (The signed mean would let pushes up and down cancel out.)
    """
    items = [{"label": label, "value": float(shap_rows[k].abs().mean())} for k, label in FEATURES.items()]
    items.sort(key=lambda i: i["value"], reverse=True)
    return items
