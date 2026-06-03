#!/usr/bin/env python3
"""
Level 3 v2 — Step 26: Transform raw similarities into a 0–100 fit score.

Reads the raw per-pair similarities (script 24) and applies:

  1. ANCHOR-LINEAR transform of each cosine to a 0–100 sub-score:

         sub_score = clip( (cos - FLOOR) / (CEIL - FLOOR), 0, 1 ) * 100

     FLOOR = 0.30  — empirical off-domain floor (the synthetic "space nurse"
                     JD against any CV averages ≈ 0.31 cosine; nothing real
                     scores below this on generic professional language).
     CEIL  = 0.80  — empirical ceiling (a CV against a JD that mirrors its
                     own most-recent role averages ≈ 0.81 cosine).
     Both skills and experience share this window — measured to be the same
     for both constructs, so one rule covers both.

  2. WEIGHTED aggregation (Level-2 reference weights):

         skills 0.6 / experience 0.3 / education 0.1

     Education edge case — if the JD states NO degree requirement
     (`jd_required_degree == "none"`), the 0.1 education weight is
     redistributed to skills and experience IN THEIR OWN PROPORTION
     (0.6 : 0.3 → skills 2/3, experience 1/3):

         w_skills     = 0.6 + 0.1 * (0.6 / 0.9) = 0.6667
         w_experience = 0.3 + 0.1 * (0.3 / 0.9) = 0.3333

     If the JD DOES require a degree, education contributes
     0.1 * education_match * 100  (i.e. +10 if the CV meets the
     requirement, 0 if it does not).

  3. Final clip(0, 100) as a hard guard (redundant given weights sum to 1
     and each sub-score is already in [0, 100], but kept per spec).

Reads:
  - results/level3/segments_normalisation/pair_similarities.csv

Writes:
  - results/level3/segments_normalisation/pair_scores.csv

Run:
    python3 scripts/level3/26_apply_scoring.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT    = Path(__file__).resolve().parent.parent.parent
IN_CSV  = ROOT / "results" / "level3" / "segments_normalisation" / "pair_similarities.csv"
OUT_CSV = ROOT / "results" / "level3" / "segments_normalisation" / "pair_scores.csv"

# Anchor-linear transform window (shared by skills + experience).
FLOOR = 0.30
CEIL  = 0.80

# Base Level-2 weights.
W_SKILLS = 0.6
W_EXP    = 0.3
W_EDU    = 0.1


def anchor_linear(cos: float) -> float:
    """Map a cosine to a 0–100 sub-score, clipped to [0, 100]."""
    if pd.isna(cos):
        return float("nan")
    return float(np.clip((cos - FLOOR) / (CEIL - FLOOR), 0.0, 1.0) * 100.0)


def score_row(row: pd.Series) -> pd.Series:
    skills_score = anchor_linear(row["skills_sim"])
    exp_score    = anchor_linear(row["experience_sim"])

    edu_required = str(row["jd_required_degree"]).lower() != "none"
    edu_match    = int(row["education_match"])

    if edu_required:
        # Education counts as its own 0.1 weight.
        w_sk, w_ex = W_SKILLS, W_EXP
        edu_contrib = W_EDU * edu_match * 100.0
        edu_score   = edu_match * 100.0          # for transparency in the CSV
        redistributed = False
    else:
        # No requirement → push education's 0.1 onto skills + experience,
        # keeping their 0.6 : 0.3 ratio.
        total = W_SKILLS + W_EXP                 # 0.9
        w_sk = W_SKILLS + W_EDU * (W_SKILLS / total)   # 0.6667
        w_ex = W_EXP    + W_EDU * (W_EXP    / total)   # 0.3333
        edu_contrib = 0.0
        edu_score   = float("nan")               # not applicable
        redistributed = True

    if pd.isna(skills_score) or pd.isna(exp_score):
        fit = float("nan")
    else:
        fit = w_sk * skills_score + w_ex * exp_score + edu_contrib
        fit = float(np.clip(fit, 0.0, 100.0))    # hard guard

    return pd.Series({
        "skills_score":        round(skills_score, 1) if not pd.isna(skills_score) else np.nan,
        "experience_score":    round(exp_score, 1)    if not pd.isna(exp_score)    else np.nan,
        "education_score":     round(edu_score, 1)    if not pd.isna(edu_score)    else np.nan,
        "edu_redistributed":   redistributed,
        "w_skills":            round(w_sk, 4),
        "w_experience":        round(w_ex, 4),
        "fit_score":           round(fit, 1) if not pd.isna(fit) else np.nan,
    })


def main() -> None:
    df = pd.read_csv(IN_CSV)
    scored = df.join(df.apply(score_row, axis=1))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT_CSV, index=False)
    print(f"✓ {len(scored)} pairs  →  {OUT_CSV.relative_to(ROOT)}\n")

    print(f"Transform: cos → clip((cos − {FLOOR})/({CEIL}−{FLOOR}), 0, 1) × 100")
    print(f"Weights:   skills {W_SKILLS} / experience {W_EXP} / education {W_EDU}")
    print(f"           (no edu requirement → skills {W_SKILLS + W_EDU*W_SKILLS/0.9:.4f} "
          f"/ experience {W_EXP + W_EDU*W_EXP/0.9:.4f})")
    print()

    real = scored[~scored.jd_id.str.startswith(("max_cv", "min_space"))]
    print(f"fit_score over {len(real)} real pairs: "
          f"min={real.fit_score.min():.0f}  "
          f"median={real.fit_score.median():.0f}  "
          f"mean={real.fit_score.mean():.0f}  "
          f"max={real.fit_score.max():.0f}")
    n_redis = int(real["edu_redistributed"].sum())
    print(f"education redistributed (no requirement) on {n_redis}/{len(real)} pairs")


if __name__ == "__main__":
    main()
