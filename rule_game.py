"""Your rule vs the model: the logic behind the first tab.

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
    "Previous_DNAs": "Most missed appointments before",
    "Deprivation_Decile": "Most deprived areas",
    "Appointment Type": "First appointments, not follow ups",
    "Specialty": "Specialities with the highest miss rate",
    "Reporting_Site": "Sites with the highest miss rate",
}
# Short names for the say chart
CLUE_SHORT = {
    "Previous_DNAs": "Missed before",
    "Age": "Age",
    "Lead_Time_Days": "Time since booking",
    "Day_Of_Week": "Day of the week",
    "Deprivation_Decile": "Deprivation",
    "Appointment Type": "New or follow-up",
    "Specialty": "Specialty",
    "Reporting_Site": "Site",
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


def nobody(past: pd.DataFrame) -> dict:
    """Phone nobody: the 'everyone attends' rule, with the best accuracy of all."""
    y = past["Did_Not_Attend"].to_numpy()
    n_miss = round(y.mean() * 1000)
    return {
        "outcomes": {"called_dna": 0, "called_attended": 0, "not_called_dna": n_miss,
                     "not_called_attended": 1000 - n_miss},
        "reached": 0.0,
        "accuracy": float((y == 0).mean()),
    }


def all_rule_catches(past: pd.DataFrame) -> dict:
    """Missed appointments on the list (per 1,000) for every rule the visitor can make:
    each combination of one or more clues, keyed by its sorted clues. 255 for 8 clues."""
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
