"""Train the DNA model on made-up past appointments, explain its forecasts with SHAP and
export everything the app needs (offline build step; the app itself needs no ML packages).

Pipeline follows lambda_ml_streamlit_exercise/build/train_and_explain.py: XGBoost with early
stopping, 75/10/15 train/validation/test split, TreeExplainer on predicted probability with
interventional perturbation, SHAP values scaled to percentage points.

Run from the repo root:  python build/train_and_explain.py
Writes data/forecast.csv, data/shap_values.csv, data/past_predictions.csv and
data/model_meta.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from demo_data import CATEGORIES, DATA_DIR, FEATURES, generate_appointments  # noqa: E402

SEED = 42  # seed for the future appointments the app shows
HISTORY_SEED = 7  # seed for the past appointments the model learns from
N_FUTURE = 3200
N_HISTORY = 30000
WEEKS = 6
N_BACKGROUND = 500  # rows used as "the average appointment" for SHAP

FORECAST_COLUMNS = [
    "row_id", "day_index", "Reporting_Site", "Specialty", "Appointment Type",
    "Appointment Location Text", "Day_Of_Week", "Previous_DNAs", "Lead_Time_Days", "Age",
    "Deprivation_Decile", "MRN", "Mobile Telephone", "Fin Number",
    "DNA_Prob_Forecast", "Confidence_Score", "DNA_Forecast",
]


def confidence_score(p, rng):
    """Synthetic stand-in for the live model's confidence score: how far the risk is
    from a coin toss, plus a little noise."""
    return np.clip(np.maximum(p, 1 - p) + rng.normal(0, 0.05, len(p)), 0.5, 0.99).round(3)


def build_matrix(df):
    """Return (X, group_of_column). Categoricals are one-hot encoded over fixed categories,
    so past and future matrices always have the same columns."""
    X = df[[c for c in FEATURES if c not in CATEGORIES]].copy()
    group = {c: c for c in X.columns}
    for c, cats in CATEGORIES.items():
        d = pd.get_dummies(pd.Categorical(df[c], categories=cats), prefix=c, prefix_sep="=", dtype=int)
        d.index = df.index
        for col in d.columns:
            group[col] = c
        X = pd.concat([X, d], axis=1)
    return X, group


def main():
    rng = np.random.default_rng(SEED)

    # ---- Past appointments, with outcomes drawn from the hidden true chance
    hist = generate_appointments(N_HISTORY, HISTORY_SEED, WEEKS)
    y = pd.Series(rng.binomial(1, hist["p_true"]), index=hist.index)
    X, group = build_matrix(hist)
    idx_train, idx_temp = train_test_split(hist.index, train_size=0.75, stratify=y, random_state=SEED)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.6, stratify=y[idx_temp], random_state=SEED)

    model = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, early_stopping_rounds=50,
                          eval_metric="logloss", enable_categorical=False, random_state=SEED)
    model.fit(X.loc[idx_train], y[idx_train], eval_set=[(X.loc[idx_val], y[idx_val])], verbose=False)
    auc = float(roc_auc_score(y[idx_test], model.predict_proba(X.loc[idx_test])[:, 1]))
    print(f"DNA rate in past appointments: {y.mean():.1%}; test AUC {auc:.3f}")

    # ---- Future appointments: the model's forecasts
    fut = generate_appointments(N_FUTURE, SEED, WEEKS)
    Xf, _ = build_matrix(fut)
    assert list(Xf.columns) == list(X.columns)
    p = model.predict_proba(Xf)[:, 1]
    fut["DNA_Prob_Forecast"] = p.round(4)
    fut["Confidence_Score"] = confidence_score(p, rng)
    fut["DNA_Forecast"] = (p >= 0.5).astype(int)

    # ---- Past appointments the model didn't train on, with what really happened, for the
    # "Who would we phone?" tab. Own rng, so the forecast above never changes because of it.
    p_past = model.predict_proba(X.loc[idx_test])[:, 1]
    past = pd.DataFrame({
        "DNA_Prob_Forecast": p_past.round(4),
        "Confidence_Score": confidence_score(p_past, np.random.default_rng(SEED + 1)),
        "Did_Not_Attend": y[idx_test].to_numpy(),
    })

    # ---- SHAP on predicted probability, one-hot columns summed back to their feature
    background = shap.maskers.Independent(
        shap.sample(X.loc[idx_train], N_BACKGROUND, random_state=SEED), max_samples=N_BACKGROUND
    )
    explainer = shap.TreeExplainer(model, data=background, model_output="probability",
                                   feature_perturbation="interventional")
    ex = explainer(Xf)
    vals = pd.DataFrame(ex.values * 100, columns=Xf.columns, index=Xf.index)
    grouped = vals.T.groupby(group).sum().T[list(FEATURES)]
    base = float(np.mean(ex.base_values) * 100)

    # ---- Checks
    err = float(np.abs(grouped.sum(axis=1).to_numpy() + base - p * 100).max())
    print(f"SHAP additivity max error {err:.4f} percentage points; base value {base:.2f}%")
    assert err < 0.5, f"SHAP does not sum to predicted probability (max err {err:.3f} pp)"

    share = fut["DNA_Forecast"].mean()
    bands = pd.cut(fut["DNA_Prob_Forecast"], [0, 0.15, 0.5, 0.8, 1], right=False,
                   labels=["Low", "Medium", "High", "Very High"]).value_counts().sort_index()
    high_risk = (fut["DNA_Prob_Forecast"] > 0.7) & (fut["Confidence_Score"] > 0.7)
    by_day = high_risk.groupby(fut["day_index"]).mean()
    print(f"Forecast as DNA: {share:.1%}")
    print("Risk bands:", bands.to_dict())
    print(f"High-risk rate by day: min {by_day.min():.1%}, median {by_day.median():.1%}, max {by_day.max():.1%}")
    print("Mean |SHAP| (pp):", grouped.abs().mean().round(2).sort_values(ascending=False).to_dict())
    assert 0.15 <= share <= 0.30, "forecast DNA share has drifted; retune demo_data.generate_appointments"
    assert (bands > 0).all(), "every risk band should have some appointments"

    # ---- Export
    DATA_DIR.mkdir(exist_ok=True)
    fut[FORECAST_COLUMNS].to_csv(DATA_DIR / "forecast.csv", index=False)
    past.to_csv(DATA_DIR / "past_predictions.csv", index=False)
    grouped.round(4).rename_axis(None).assign(row_id=fut["row_id"].to_numpy())[["row_id", *FEATURES]].to_csv(
        DATA_DIR / "shap_values.csv", index=False
    )
    meta = {
        "base_value_pp": base,
        "auc_test": auc,
        "dna_rate_history": float(y.mean()),
        "n_history": N_HISTORY,
        "n_future": N_FUTURE,
        "n_past_test": len(past),
        "dna_rate_past_test": float(past["Did_Not_Attend"].mean()),
        "seed": SEED,
        "history_seed": HISTORY_SEED,
        "shap_additivity_max_error_pp": err,
        "versions": {"xgboost": xgboost.__version__, "shap": shap.__version__,
                     "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (DATA_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {DATA_DIR}")


if __name__ == "__main__":
    main()
