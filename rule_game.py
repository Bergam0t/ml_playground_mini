"""Bonus: beat the model: the logic behind the game on the last tab.

A visitor picks clues to phone by; each clue they pick gets an equal say. Their rule is
scored on the 4,500 past appointments the model didn't learn from, at the same number of
calls as the model gets. The model's results, learning from more and more past
appointments, are precomputed by build/rule_game_data.py (the app needs no ML packages).
"""
import itertools
import json

import numpy as np
import pandas as pd

import calls
from demo_data import DATA_DIR

# Share of appointments the team can phone: 100 calls per 1,000 appointments
CALL_SHARE = 0.10

# Clue -> button label. The direction is given: the visitor chooses which clues to use and
# the rule gives each an equal say, so the game is about how much each clue should count.
CLUES = {
    "Age": "Being younger",
    "Lead_Time_Days": "A longer time since booking",
    "Day_Of_Week": "Riskiest days of the week",
    "Time_Slot": "Riskiest times of day",
    "Previous_DNAs": "Most missed appointments before",
    "Deprivation_Decile": "Most deprived areas",
    "Appointment Type": "First appointments, not follow ups",
    "Specialty": "Specialities with the highest miss rate",
    "Reporting_Site": "Sites with the highest miss rate",
    "Journey_Time_Mins": "Longer journeys",
}
# Short names for the say chart
CLUE_SHORT = {
    "Previous_DNAs": "Missed before",
    "Age": "Age",
    "Lead_Time_Days": "Time since booking",
    "Day_Of_Week": "Day of the week",
    "Time_Slot": "Time of day",
    "Deprivation_Decile": "Deprivation",
    "Appointment Type": "New or follow-up",
    "Specialty": "Specialty",
    "Reporting_Site": "Site",
    "Journey_Time_Mins": "Journey time",
}


def load_past():
    return pd.read_csv(DATA_DIR / "rule_game_past.csv")


def load_learning():
    return json.loads((DATA_DIR / "rule_game_learning.json").read_text(encoding="utf-8"))


def rule_score(past: pd.DataFrame, clues) -> np.ndarray:
    """Equal say for each clue: add up where each appointment ranks on it (0 to 1).
    Ties, common for yes/no clues, are broken at random but the same way every time."""
    total = sum(past[c].rank(pct=True) for c in clues)
    jitter = np.random.default_rng(0).random(len(past)) * 1e-6
    return np.asarray(total, dtype=float) + jitter


def results(past: pd.DataFrame, score) -> dict:
    """Outcomes per 1,000 (for charts.outcome_blocks), share of misses reached and accuracy,
    when phoning the top CALL_SHARE by score. Matches results() in build/rule_game_data.py."""
    y = past["Did_Not_Attend"].to_numpy()
    called = calls.called_by_share(score, CALL_SHARE)
    out = calls.call_outcomes(called, y, score)
    return {
        "outcomes": {k: out[k] for k, _ in calls.GROUPS},
        "reached": float((called & (y == 1)).sum() / y.sum()),
        "accuracy": float(((called & (y == 1)) | (~called & (y == 0))).mean()),
    }


# The simple rule a manager would reach for first: phone the people who've missed most before.
# It is also the best rule the game's buttons can make.
SIMPLE_RULE = ("Previous_DNAs",)


def precision_at_share(past: pd.DataFrame, score, share: float) -> float:
    """Share of calls that reach someone who then missed, phoning the top `share` by score."""
    called = calls.called_by_share(score, share)
    return float(past["Did_Not_Attend"].to_numpy()[called].mean()) if called.any() else 0.0


def three_ways(past: pd.DataFrame, model_risk) -> dict:
    """Missed appointments reached per 1,000 past appointments with CALL_SHARE calls: at
    random, by SIMPLE_RULE and by the model. `model_risk` is the live model's predicted risk
    for the same appointments, in the same order (data/past_predictions.csv)."""
    n_missed = round(past["Did_Not_Attend"].mean() * 1000)
    return {
        "n_missed": n_missed,
        "n_calls": round(CALL_SHARE * 1000),
        "random": n_missed * CALL_SHARE,
        "rule": results(past, rule_score(past, SIMPLE_RULE))["outcomes"]["called_dna"],
        "model": results(past, np.asarray(model_risk, dtype=float))["outcomes"]["called_dna"],
    }


def all_rule_catches(past: pd.DataFrame) -> dict:
    """Missed appointments on the list (per 1,000) for every rule the visitor can make:
    each combination of one or more clues, keyed by its sorted clues. 1,023 for 10 clues."""
    return {
        tuple(sorted(combo)): results(past, rule_score(past, combo))["outcomes"]["called_dna"]
        for k in range(1, len(CLUES) + 1)
        for combo in itertools.combinations(CLUES, k)
    }


def rank_of(caught: int, catches: dict) -> int:
    """Where a rule that catches `caught` ranks among all the rules: 1 + those that catch more."""
    return 1 + sum(c > caught for c in catches.values())


def overtook_at(frames: list, score: float):
    """The first frame (index) where the model beats `score`, or None."""
    for i, f in enumerate(frames):
        if f["reached"] > score:
            return i
    return None
