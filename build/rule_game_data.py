"""Data for the "Your rule vs the model" tab (offline build step, like train_and_explain.py).

The tab lets a visitor build a rule for who to phone from the eight clues the model sees,
each clue with an equal say. Then it shows a computer learning how much each clue should
count, from more and more past appointments, and overtaking the visitor's rule.

Writes:
- data/rule_game_past.csv: the 4,500 past test appointments (the same ones as
  data/past_predictions.csv), with what really happened, the hidden true chance and one
  score per clue that a person could phone by (higher = phone first).
- data/rule_game_learning.json: the computer's results after learning from 0, 25, 50, ...
  22,500 past appointments: outcomes per 1,000 at the tab's call budget, and how much say
  each clue ends up with.

The computer is a logistic regression on the same eight clues: the simplest model that
learns how much each clue counts. It has to learn which way each clue points too (raw
inputs, categories one-hot), so it starts out worse than a sensible person. On these past
appointments it reaches the same share of misses as the app's XGBoost model.

Run from the repo root:  python build/rule_game_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))
import calls  # noqa: E402
from demo_data import CATEGORIES, DATA_DIR, FEATURES  # noqa: E402
from rule_game import CALL_SHARE  # noqa: E402
from train_and_explain import build_matrix, load_history  # noqa: E402

# Training-set sizes the computer learns from, one animation frame each (0 = no data yet,
# so its picks are random). Nested: each size adds appointments to the one before.
SIZES = [0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 22500]
# Order the training appointments are fed in. Checked with seeds 1-3: below about 400
# appointments the curve depends on which appointments come first, above it barely does.
ORDER_SEED = 3


def clue_scores(train_df, y_train, df):
    """One 'phone first' score per clue, as a sensible person would use it: numbers in the
    direction that goes with more missed appointments, categories by last year's miss rate."""
    out = {}
    for f in FEATURES:
        if f in CATEGORIES:
            out[f] = df[f].map(y_train.groupby(train_df[f]).mean()).astype(float)
        else:
            sign = np.sign(np.corrcoef(train_df[f], y_train)[0, 1])
            out[f] = sign * df[f].astype(float)
    return pd.DataFrame(out, index=df.index)


def results(score, y, rng):
    """Outcomes per 1,000 when phoning the top CALL_SHARE by score; ties broken at random."""
    jitter = rng.random(len(y)) * 1e-9 * (np.ptp(score) or 1)
    called = calls.called_by_share(np.asarray(score, dtype=float) + jitter, CALL_SHARE)
    y = np.asarray(y)
    out = calls.call_outcomes(called, y, np.asarray(score, dtype=float))
    return {
        "outcomes": {k: out[k] for k, _ in calls.GROUPS},
        "reached": float((called & (y == 1)).sum() / y.sum()),
        "accuracy": float(((called & (y == 1)) | (~called & (y == 0))).mean()),
    }


def main():
    hist, y, idx_train, _, idx_test = load_history()
    train, test = hist.loc[idx_train], hist.loc[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    past = pd.read_csv(DATA_DIR / "past_predictions.csv")
    assert (past["Did_Not_Attend"].to_numpy() == y_test.to_numpy()).all(), \
        "test split differs from data/past_predictions.csv: rerun build/train_and_explain.py"

    # ---- The visitor's clues
    scores = clue_scores(train, y_train, test).reset_index(drop=True)
    scores.insert(0, "p_true", test["p_true"].round(5).to_numpy())
    scores.insert(0, "Did_Not_Attend", y_test.to_numpy())

    # ---- The computer: raw inputs, so it must learn each clue's direction as well as its say
    X, group = build_matrix(hist)
    X["Lead_Time_Days"] = np.log(X["Lead_Time_Days"])
    scaler = StandardScaler().fit(X.loc[idx_train])
    X_tr = scaler.transform(X.loc[idx_train])
    X_te = scaler.transform(X.loc[idx_test])
    cols = {f: [i for i, c in enumerate(X.columns) if group[c] == f] for f in FEATURES}

    order = np.random.default_rng(ORDER_SEED).permutation(len(idx_train))
    y_tr = y_train.to_numpy()
    frames = []
    for n in SIZES:
        rng = np.random.default_rng(0)
        if n == 0:
            score, say = np.zeros(len(y_test)), {f: 0.0 for f in FEATURES}
        else:
            idx = order[:n]
            model = LogisticRegression(C=1.0, max_iter=5000).fit(X_tr[idx], y_tr[idx])
            score = model.decision_function(X_te)
            # A clue's say: how much its part of the score varies across appointments
            spread = {f: float(np.std(X_te[:, c] @ model.coef_[0, c])) for f, c in cols.items()}
            total = sum(spread.values())
            say = {f: v / total for f, v in spread.items()}
        frames.append({"n": n, "misses_seen": int(y_tr[order[:n]].sum()), "say": say,
                       **results(score, y_test, rng)})
        print(f"learnt from {n:6,}: reaches {frames[-1]['reached']:.0%} of misses")

    xgb = results(past["DNA_Prob_Forecast"], y_test, np.random.default_rng(0))
    perfect = results(test["p_true"], y_test, np.random.default_rng(0))
    print(f"app's XGBoost model: {xgb['reached']:.0%}; perfect model: {perfect['reached']:.0%}")
    assert abs(frames[-1]["reached"] - xgb["reached"]) < 0.03, "the computer no longer matches the app's model"

    scores.round(5).to_csv(DATA_DIR / "rule_game_past.csv", index=False)
    (DATA_DIR / "rule_game_learning.json").write_text(json.dumps({
        "call_share": CALL_SHARE,
        "n_train": len(idx_train),
        "frames": frames,
        "app_model": xgb,
        "perfect": perfect,
    }, indent=1))
    print(f"Wrote {DATA_DIR / 'rule_game_past.csv'} and rule_game_learning.json")


if __name__ == "__main__":
    main()
