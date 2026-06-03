#!/usr/bin/env python3
"""
Level 3 v2 — Step 25: Cross-level comparison table.

Merge — without any weighted formula — the raw signals from all three levels
into one row per (CV, JD) pair, so the user can eyeball correlation between
L1 holistic, L2 categorical + computed fit, and L3v2 block-similarity.

Aggregation across the 3 runs per pair:
  L1: mean of `score`, `score_100` (P2 only) per prompt.
  L2: mode of categorical labels, mean of numeric sub-scores + holistic.

L3 v2 is deterministic (one row per pair, no runs).

Anchor JDs (max_cv_*, min_space_nurse) are excluded automatically because
they have no L1/L2 coverage.

Reads:
  - results/level1_master.csv
  - results/level2_master.csv
  - results/level3/segments_normalisation/pair_similarities.csv

Writes:
  - results/level3/segments_normalisation/all_levels_comparison.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT     = Path(__file__).resolve().parent.parent.parent
L1_CSV   = ROOT / "results" / "level1_master.csv"
L2_CSV   = ROOT / "results" / "level2_master.csv"
L3_CSV   = ROOT / "results" / "level3" / "segments_normalisation" / "pair_similarities.csv"
OUT_CSV  = ROOT / "results" / "level3" / "segments_normalisation" / "all_levels_comparison.csv"


def mode_label(s: pd.Series):
    s = s.dropna()
    return s.value_counts().idxmax() if len(s) else None


# ──────────────────────────────────────────────────────────────────────────
# Level-1 aggregation
# ──────────────────────────────────────────────────────────────────────────
def aggregate_l1(l1: pd.DataFrame) -> pd.DataFrame:
    """Pivot L1 into one row per (cv, jd_id) with per-prompt mean scores.
    Only P1 and P2 are kept (P0 also exists but the user explicitly asked
    for P1 and P2 in earlier comparison work)."""
    df = l1[l1["error"].isna()].copy()
    df["score"]     = df["score"].astype(float)
    df["score_100"] = df["score_100"].astype(float)

    piv = (df.pivot_table(index=["cv", "jd_id"],
                          columns="prompt",
                          values=["score", "score_100"],
                          aggfunc="mean"))
    piv.columns = [f"L1_{p}_{kind}" for kind, p in piv.columns]
    rename = {
        "L1_P1_score":     "L1_P1_score_0_3",
        "L1_P2_score":     "L1_P2_score_0_3",
        "L1_P2_score_100": "L1_P2_score_100",
        # P0 / P1 score_100 don't exist; drop the columns silently if pivot made them.
    }
    piv = piv.rename(columns=rename)
    keep = [c for c in
            ("L1_P1_score_0_3", "L1_P2_score_0_3", "L1_P2_score_100")
            if c in piv.columns]
    piv = piv[keep].reset_index().rename(columns={"cv": "cv_id"})
    for c in keep:
        piv[c] = piv[c].round(2)
    return piv


# ──────────────────────────────────────────────────────────────────────────
# Level-2 aggregation (matches build_level2_master.py formula for recompute)
# ──────────────────────────────────────────────────────────────────────────
W_SKILL, W_EXP, W_EDU = 0.6, 0.3, 0.1   # L2 reference weights (for ref column)


def aggregate_l2(l2: pd.DataFrame) -> pd.DataFrame:
    df = l2[l2["error"].isna()].copy()
    cat = ["holistic"]
    num = ["holistic_score", "fit_score_100",
           "skill_score", "role_score", "domain_score", "edu_score"]
    agg = {c: mode_label for c in cat}
    agg.update({c: "mean" for c in num})

    g = (df.groupby(["cv", "jd_id", "source"])
           .agg(agg)
           .reset_index()
           .rename(columns={"cv": "cv_id", "source": "jd_pool"}))

    # Experience score uses the same 0.4 / 0.6 role-domain split as L2 main formula.
    g["experience_score"] = 0.4 * g["role_score"] + 0.6 * g["domain_score"]

    # Recomputed weighted score from the averaged sub-scores
    # (slightly differs from L2_fit_score_100 mean by Jensen's inequality —
    #  exposing the weighting on the page rather than hiding it inside fit_score_100).
    g["weighted_from_subscales_100"] = 100.0 * (
        W_SKILL * g["skill_score"]
      + W_EXP   * g["experience_score"]
      + W_EDU   * g["edu_score"]
    )

    rename = {
        "holistic":                  "L2_holistic",
        "holistic_score":            "L2_holistic_score",
        "fit_score_100":             "L2_fit_score_100",
        "skill_score":               "L2_skill_score",
        "role_score":                "L2_role_score",
        "domain_score":              "L2_domain_score",
        "edu_score":                 "L2_edu_score",
        "experience_score":          "L2_experience_score",
        "weighted_from_subscales_100": "L2_weighted_from_subscales_100",
    }
    g = g.rename(columns=rename)
    for c in ["L2_holistic_score", "L2_fit_score_100",
              "L2_skill_score", "L2_role_score", "L2_domain_score",
              "L2_experience_score", "L2_edu_score",
              "L2_weighted_from_subscales_100"]:
        g[c] = g[c].round(3)
    return g


# ──────────────────────────────────────────────────────────────────────────
# Level-3 v2: load the already-computed pair similarities + rename columns
# ──────────────────────────────────────────────────────────────────────────
def load_l3v2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {
        "skills_sim":         "L3v2_skills_sim",
        "experience_sim":     "L3v2_experience_sim",
        "education_match":    "L3v2_education_match",
        "cv_highest_degree":  "L3v2_cv_highest_degree",
        "jd_required_degree": "L3v2_jd_required_degree",
        "n_skills_cv":        "L3v2_n_skills_cv",
        "n_experience_cv":    "L3v2_n_experience_cv",
        "n_skills_jd":        "L3v2_n_skills_jd",
        "n_experience_jd":    "L3v2_n_experience_jd",
    }
    return df.rename(columns=rename)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    l1_agg = aggregate_l1(pd.read_csv(L1_CSV))
    l2_agg = aggregate_l2(pd.read_csv(L2_CSV))
    l3_agg = load_l3v2(L3_CSV)

    # Inner join on L2 ∩ L3 (so anchor JDs and any docs L2 doesn't cover drop out).
    # L1 left-joined because P1 only covers `main`.
    df = (l3_agg
          .merge(l2_agg, on=["cv_id", "jd_id"], how="inner")
          .merge(l1_agg, on=["cv_id", "jd_id"], how="left"))

    col_order = [
        "cv_id", "jd_id", "jd_pool",
        # ── Level 1 ─────────────────────────────────────────
        "L1_P1_score_0_3", "L1_P2_score_0_3", "L1_P2_score_100",
        # ── Level 2 ─────────────────────────────────────────
        "L2_holistic", "L2_holistic_score", "L2_fit_score_100",
        "L2_skill_score", "L2_role_score", "L2_domain_score",
        "L2_experience_score", "L2_edu_score",
        "L2_weighted_from_subscales_100",
        # ── Level 3 v2 (no aggregation yet) ─────────────────
        "L3v2_skills_sim", "L3v2_experience_sim", "L3v2_education_match",
        "L3v2_cv_highest_degree", "L3v2_jd_required_degree",
        "L3v2_n_skills_cv", "L3v2_n_experience_cv",
        "L3v2_n_skills_jd", "L3v2_n_experience_jd",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["cv_id", "L3v2_skills_sim"],
                        ascending=[True, False]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ {len(df)} pairs  →  {OUT_CSV.relative_to(ROOT)}\n")

    # ── Per-CV summary of headline scores ─────────────────────────────────
    headline = ["L1_P2_score_100", "L2_fit_score_100",
                "L2_weighted_from_subscales_100",
                "L3v2_skills_sim", "L3v2_experience_sim"]
    print("Per-CV mean of headline columns:")
    print(df.groupby("cv_id")[headline].mean().round(3).to_string())
    print()

    # ── Pool × CV breakdown for L3v2 skills_sim ──────────────────────────
    print("L3v2_skills_sim mean by (cv_id × jd_pool):")
    print(df.pivot_table(index="cv_id", columns="jd_pool",
                        values="L3v2_skills_sim", aggfunc="mean")
            .round(3).to_string())
    print()
    print("L3v2_experience_sim mean by (cv_id × jd_pool):")
    print(df.pivot_table(index="cv_id", columns="jd_pool",
                        values="L3v2_experience_sim", aggfunc="mean")
            .round(3).to_string())
    print()
    print(f"L3v2_education_match (% = 1) by jd_pool:")
    print((df.groupby("jd_pool")["L3v2_education_match"].mean() * 100)
            .round(1).to_string())
    print()

    # ── Pearson + Spearman between headline 0-100 scores ─────────────────
    # (L2 weighted and L2 fit are essentially identical so we drop the latter
    #  to keep the matrix readable.)
    corr_cols = ["L1_P2_score_100", "L2_weighted_from_subscales_100",
                 "L3v2_skills_sim", "L3v2_experience_sim"]
    sub = df[corr_cols].dropna(how="any")
    print(f"Pearson r — n = {len(sub)} pairs (rows with all 4 columns present):")
    print(sub.corr(method="pearson").round(3).to_string())
    print()
    print("Spearman ρ on the same subset:")
    print(sub.corr(method="spearman").round(3).to_string())


if __name__ == "__main__":
    main()
