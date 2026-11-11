"""How many people should we phone? Choose appointments to call and count what happens, per 1,000.

Adapted from metrics.call_outcomes() in lambda_ml_streamlit_exercise. The outcomes come
from past appointments the model didn't train on (data/past_predictions.csv), so we know
who really did not attend. Nobody in them was phoned: "would get a call" is what the
model's list would have said, and "in reality" is what happened without a call.
"""
import numpy as np

GROUPS = [
    ("called_dna", "Would get a call, and in reality did not attend"),
    ("called_attended", "Would get a call, but in reality attended anyway"),
    ("not_called_dna", "Would not get a call, but in reality did not attend"),
    ("not_called_attended", "Would not get a call, and attended"),
]


def called_by_share(p, share: float) -> np.ndarray:
    """Call the highest-risk `share` of appointments. Ties are broken by a stable sort."""
    p = np.asarray(p, dtype=float)
    n_called = int(round(len(p) * float(np.clip(share, 0, 1))))
    called = np.zeros(len(p), dtype=bool)
    called[np.argsort(-p, kind="stable")[:n_called]] = True
    return called


def called_by_threshold(p, risk_min: float) -> np.ndarray:
    """Call every appointment with a predicted risk at or above the minimum."""
    return np.asarray(p, dtype=float) >= risk_min


def call_outcomes(called, y, p) -> dict:
    """Scale the calls and outcomes to 1,000 appointments, rounded so the four counts add
    up to exactly 1,000. These are appointments that WERE missed, not DNAs a call prevented."""
    called = np.asarray(called, dtype=bool)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)

    calls = int(round(called.sum() * 1000 / n))
    would_miss = int(round(y.mean() * 1000))
    called_dna = int(round((called & (y == 1)).sum() * 1000 / n))
    called_dna = min(called_dna, calls, would_miss)
    out = {
        "called_dna": called_dna,
        "called_attended": calls - called_dna,
        "not_called_dna": would_miss - called_dna,
        "not_called_attended": 1000 - calls - (would_miss - called_dna),
    }
    assert sum(out.values()) == 1000 and min(out.values()) >= 0
    out.update(
        calls=calls,
        dna_total=would_miss,
        cutoff=float(p[called].min()) if called.any() else None,
        precision=called_dna / calls if calls else None,
        recall=called_dna / would_miss if would_miss else None,
        random_precision=would_miss / 1000,
    )
    return out
