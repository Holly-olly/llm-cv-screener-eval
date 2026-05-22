#!/usr/bin/env python3
"""
Level 3 — Step 20: Score transformation. Convert raw MaxSim per-block
cosine similarities into a 0–100 fit score that is comparable across
constructs and across (CV, JD) pairs.

The cosine block values from `maxsim_summary.csv` live in DIFFERENT
empirical ranges per construct (skills narrower, experience wider).
Applying the Level-2 weighting formula directly to raw cosines biases
the result against skills. The fix is a two-step transformation:

  Step 1 — per-construct LINEAR-ADJ normalisation to [0, 1] using the
  corpus min and max anchors from `anchors_summary.csv`, with the upper
  anchor shrunk by ADJ_FACTOR (= 0.8) of the (max − min) range:

      adj_hi = lo + ADJ_FACTOR · (hi - lo)
      norm(x, lo, adj_hi) = clip( (x - lo) / (adj_hi - lo), 0, 1 )

  The shrinkage acknowledges that the synthetic max anchor (an LLM-
  generated CV-mirroring JD) sits at the upper EDGE of the cosine
  geometry, not at the centre of the "strong fit" band. Treating 80 %
  of the anchor range as the realistic ceiling lets real strong-fit
  JDs reach values near 1.0 after normalisation.

  Step 2 — Level-2 weighted average on the normalised sub-scores:

      fit_score_0_100 = 100 × ( 0.6 · skills_norm
                              + 0.3 · experience_norm
                              + 0.1 · education_norm )

Education edge cases are inherited from Method B raw:
  - JD has no education requirement → `education_block` already holds
    the fallback mean(skills_block, experience_block); normalising it
    against the education anchors gives a moderate edu_norm value.
  - JD requires education AND CV has none → `education_block` is NaN;
    normalisation propagates NaN; final fit_score is NaN.

Reads:
  - results/level3/maxsim_summary.csv
  - results/level3/anchors_summary.csv

Writes:
  - results/level3/methodB_final_summary.csv

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/20_score_transformation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT        = Path(__file__).parent.parent.parent
METHODB_CSV = ROOT / "results" / "level3" / "maxsim_summary.csv"
ANCHORS_CSV = ROOT / "results" / "level3" / "anchors_summary.csv"
OUT_CSV     = ROOT / "results" / "level3" / "methodB_final_summary.csv"

CV_IDS = ["cv_primary", "cv_hr", "cv_engineer"]
MAX_ANCHOR_FOR = {
    "cv_primary":  "max_cv_primary",
    "cv_hr":       "max_cv_hr",
    "cv_engineer": "max_cv_engineer",
}
MIN_ANCHOR_NAME = "min_space_nurse"
BLOCKS = ("skills_block", "experience_block", "education_block")

# Level-2 weights — kept identical for cross-level comparability.
W_SKILLS = 0.6
W_EXP    = 0.3
W_EDU    = 0.1

# Linear-adj shrinkage. The synthetic max anchor sits at the upper EDGE of
# the cosine geometry, not at the centre of the "strong fit" band.
# Treating ADJ_FACTOR · (hi - lo) as the realistic ceiling lets real strong-
# fit JDs reach values near 1.0 after normalisation. Equivalent to:
#     norm_adj = clip(norm_base / ADJ_FACTOR, 0, 1)
ADJ_FACTOR = 0.8


def compute_block_anchors(anc: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-construct corpus anchors (mean across the 3 CVs).

    Returns the raw `min` / `max` anchor pair AND the shrunk `adj_max`
    used as the upper end of the normalisation window:
        adj_max = min + ADJ_FACTOR · (max - min)
    """
    out: dict[str, dict[str, float]] = {}
    for b in BLOCKS:
        mins, maxs = [], []
        for cv in CV_IDS:
            mn = float(anc[(anc.cv_id == cv) &
                           (anc.anchor_jd == MIN_ANCHOR_NAME)][b].iloc[0])
            mx = float(anc[(anc.cv_id == cv) &
                           (anc.anchor_jd == MAX_ANCHOR_FOR[cv])][b].iloc[0])
            mins.append(mn)
            maxs.append(mx)
        lo = float(np.mean(mins))
        hi = float(np.mean(maxs))
        out[b] = {
            "min":     lo,
            "max":     hi,
            "adj_max": lo + ADJ_FACTOR * (hi - lo),
        }
    return out


def linear_normalise(x, lo: float, hi: float):
    if pd.isna(x) or hi <= lo:
        return float("nan")
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def main() -> None:
    df  = pd.read_csv(METHODB_CSV)
    anc = pd.read_csv(ANCHORS_CSV)
    ba  = compute_block_anchors(anc)

    print(f"Per-construct corpus anchors (linear-adj, ADJ_FACTOR={ADJ_FACTOR}):")
    print(f"  {'block':<18}  {'min':>8}  {'max':>8}  {'adj_max':>9}  {'range_adj':>10}")
    for b in BLOCKS:
        print(f"  {b:<18}  {ba[b]['min']:>8.4f}  {ba[b]['max']:>8.4f}  "
              f"{ba[b]['adj_max']:>9.4f}  {ba[b]['adj_max'] - ba[b]['min']:>10.4f}")
    print(f"\nFinal-score weights: skills={W_SKILLS}, experience={W_EXP}, "
          f"education={W_EDU}")
    print()

    # Upper end of the normalisation window is the SHRUNK anchor max.
    sk_lo, sk_hi = ba["skills_block"]["min"],     ba["skills_block"]["adj_max"]
    ex_lo, ex_hi = ba["experience_block"]["min"], ba["experience_block"]["adj_max"]
    ed_lo, ed_hi = ba["education_block"]["min"],  ba["education_block"]["adj_max"]

    df["skills_norm"]     = df["skills_block"].apply(
        lambda x: linear_normalise(x, sk_lo, sk_hi))
    df["experience_norm"] = df["experience_block"].apply(
        lambda x: linear_normalise(x, ex_lo, ex_hi))
    df["education_norm"]  = df["education_block"].apply(
        lambda x: linear_normalise(x, ed_lo, ed_hi))

    def weighted(row):
        parts = [row["skills_norm"], row["experience_norm"], row["education_norm"]]
        if any(pd.isna(p) for p in parts):
            return float("nan")
        return round(100.0 * (W_SKILLS * parts[0] +
                              W_EXP    * parts[1] +
                              W_EDU    * parts[2]), 1)

    df["fit_score_0_100"] = df.apply(weighted, axis=1)

    for c in ("skills_norm", "experience_norm", "education_norm"):
        df[c] = df[c].round(4)

    base_cols = [
        "cv_id", "jd_id", "jd_pool",
        "n_cv", "n_jd",
        "n_cv_skills", "n_cv_experience", "n_cv_education", "n_cv_mixed",
        "n_jd_skills", "n_jd_experience", "n_jd_education", "n_jd_mixed",
        "skills_block",     "skills_norm",
        "experience_block", "experience_norm",
        "education_block",  "edu_required", "edu_fallback_used",
        "education_norm",
        "fit_score_0_100",
    ]
    df = df[base_cols]
    df = df.sort_values(["cv_id", "fit_score_0_100"],
                        ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Transformed scores → {OUT_CSV.relative_to(ROOT)}  ({len(df)} pairs)\n")

    show = ["jd_pool", "jd_id",
            "skills_block",     "skills_norm",
            "experience_block", "experience_norm",
            "education_block",  "education_norm",
            "fit_score_0_100"]
    for cv in CV_IDS:
        sub = df[df["cv_id"] == cv].head(5)
        print(f"── {cv} ──  (TOP 5 by fit_score_0_100)")
        with pd.option_context("display.max_colwidth", 55, "display.width", 200):
            print(sub[show].to_string(index=False))
        print()

    print("Mean fit_score_0_100 by (cv_id × jd_pool):")
    pivot = df.pivot_table(index="cv_id", columns="jd_pool",
                           values="fit_score_0_100", aggfunc="mean").round(1)
    print(pivot.to_string())
    print()

    print("Distribution of fit_score_0_100:")
    print(df["fit_score_0_100"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
