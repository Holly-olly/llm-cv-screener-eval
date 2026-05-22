#!/usr/bin/env python3
"""
Level 3 — Step 21: Merge all scores from Levels 1, 2 and 3 into one
side-by-side comparison table — one row per (CV, JD) pair.

L1 (1 prompt × 3 runs aggregated):
  - L1_P1_score_0_3       mean of 0–3 scores across P1 runs
  - L1_P2_score_0_3       mean of 0–3 scores across P2 runs
  - L1_P2_score_100       mean of 0–100 scores across P2 runs
    (P0 / P1 cover only the `main` pool — other pools come back as NaN.
     Only P2 carries a 0–100 score; P0 / P1 are 0–3 only.)

L2 (1 prompt × 3 runs aggregated):
  - L2_holistic                       mode of the 4-level label
  - L2_holistic_score                 mean of 0–3 holistic score
  - L2_fit_score_100                  mean of per-run 0–100 fit scores
  - L2_skill_score                    mean of 0–1 skills sub-score
  - L2_role_score                     mean of 0–1 role-relevance sub-score
  - L2_domain_score                   mean of 0–1 domain-relevance sub-score
  - L2_experience_score               mean of (0.4 · role + 0.6 · domain)
  - L2_edu_score                      mean of 0–1 education sub-score
  - L2_weighted_from_subscales_100    explicit recompute:
        100 × ( 0.6 · L2_skill_score
              + 0.3 · L2_experience_score
              + 0.1 · L2_edu_score )

L3 (deterministic, no runs — one number per pair):
  - L3_skills_block / experience_block / education_block   raw MaxSim
  - L3_skills_norm  / experience_norm  / education_norm    linear-adj [0, 1]
  - L3_fit_score_0_100                                     final 0–100
  - L3_edu_required, L3_edu_fallback_used                  edu edge-case flags

Reads:
  - results/level1_master.csv
  - results/level2_master.csv
  - results/level3/maxsim_summary.csv
  - results/level3/methodB_final_summary.csv

Writes:
  - results/level3/all_levels_comparison.csv

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/21_compare_l2_l3.py
"""

from pathlib import Path

import pandas as pd


ROOT     = Path(__file__).parent.parent.parent
L1_CSV   = ROOT / "results" / "level1_master.csv"
L2_CSV   = ROOT / "results" / "level2_master.csv"
L3_RAW   = ROOT / "results" / "level3" / "maxsim_summary.csv"
L3_FINAL = ROOT / "results" / "level3" / "methodB_final_summary.csv"
OUT_CSV  = ROOT / "results" / "level3" / "all_levels_comparison.csv"

# L2 weighted-from-subscales formula (matches build_level2_master.py).
W_SKILL, W_EXP, W_EDU = 0.6, 0.3, 0.1


def mode_label(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return None
    return s.value_counts().idxmax()


def aggregate_l1(l1: pd.DataFrame) -> pd.DataFrame:
    """One row per (cv, jd) with the L1 prompt-specific means."""
    l1 = l1[l1["error"].isna()].copy()
    out = (l1.assign(score=l1["score"].astype(float),
                     score_100=l1["score_100"].astype(float))
             .pivot_table(index=["cv", "jd_id"],
                          columns="prompt",
                          values=["score", "score_100"],
                          aggfunc="mean"))
    # Flatten the MultiIndex columns into the names we actually want.
    out.columns = [f"L1_{p}_score_{kind.split('_')[-1]}"
                   for kind, p in out.columns]
    # Keep only the requested columns; rename for clarity.
    rename = {
        "L1_P1_score_score": "L1_P1_score_0_3",      # P1 has only 0–3
        "L1_P2_score_score": "L1_P2_score_0_3",      # P2 has 0–3 too
        "L1_P2_score_100":   "L1_P2_score_100",      # P2 has 0–100 as well
    }
    out = out.rename(columns=rename)
    keep = [c for c in
            ("L1_P1_score_0_3", "L1_P2_score_0_3", "L1_P2_score_100")
            if c in out.columns]
    out = out[keep].reset_index().rename(columns={"cv": "cv_id"})
    for c in keep:
        out[c] = out[c].round(2)
    return out


def aggregate_l2(l2: pd.DataFrame) -> pd.DataFrame:
    """One row per (cv, jd) with L2 label modes + numeric means + recomputed
    weighted score from the per-pair mean subscales."""
    l2 = l2[l2["error"].isna()].copy()
    cat = ["holistic"]
    num = ["holistic_score", "fit_score_100",
           "skill_score", "role_score", "domain_score", "edu_score"]
    agg = {c: mode_label for c in cat}
    agg.update({c: "mean" for c in num})
    g = (l2.groupby(["cv", "jd_id"])
           .agg(agg)
           .reset_index()
           .rename(columns={"cv": "cv_id"}))

    # Experience score = same 0.4 · role + 0.6 · domain combo L2 uses internally.
    g["experience_score"] = 0.4 * g["role_score"] + 0.6 * g["domain_score"]

    # Recompute 0–100 fit explicitly from the (averaged) subscales — exposes
    # the weighting on the page rather than hiding it inside fit_score_100.
    g["weighted_from_subscales_100"] = (
        100.0 * (W_SKILL * g["skill_score"]
               + W_EXP   * g["experience_score"]
               + W_EDU   * g["edu_score"])
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


def aggregate_l3(raw: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """One row per (cv, jd) with raw MaxSim blocks + linear-adj normalised
    sub-scores + final 0–100 fit."""
    cols_raw = ["cv_id", "jd_id", "jd_pool",
                "skills_block", "experience_block", "education_block",
                "edu_required", "edu_fallback_used"]
    cols_fin = ["cv_id", "jd_id",
                "skills_norm", "experience_norm", "education_norm",
                "fit_score_0_100"]
    m = raw[cols_raw].merge(final[cols_fin], on=["cv_id", "jd_id"], how="inner")
    rename = {
        "skills_block":      "L3_skills_block",
        "experience_block":  "L3_experience_block",
        "education_block":   "L3_education_block",
        "skills_norm":       "L3_skills_norm",
        "experience_norm":   "L3_experience_norm",
        "education_norm":    "L3_education_norm",
        "edu_required":      "L3_edu_required",
        "edu_fallback_used": "L3_edu_fallback_used",
        "fit_score_0_100":   "L3_fit_score_0_100",
    }
    return m.rename(columns=rename)


def main() -> None:
    l1_agg = aggregate_l1(pd.read_csv(L1_CSV))
    l2_agg = aggregate_l2(pd.read_csv(L2_CSV))
    l3_agg = aggregate_l3(pd.read_csv(L3_RAW), pd.read_csv(L3_FINAL))

    # L3 has every pair (3 CVs × 75 JDs = 225). Outer-left-merge L1 and L2
    # onto L3 so cells without L1 P1 / P2 coverage become NaN.
    df = (l3_agg
          .merge(l2_agg, on=["cv_id", "jd_id"], how="left")
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
        # ── Level 3 ─────────────────────────────────────────
        "L3_skills_block", "L3_experience_block", "L3_education_block",
        "L3_edu_required", "L3_edu_fallback_used",
        "L3_skills_norm",  "L3_experience_norm",  "L3_education_norm",
        "L3_fit_score_0_100",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["cv_id", "L3_fit_score_0_100"],
                        ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Merged comparison → {OUT_CSV.relative_to(ROOT)}  ({len(df)} pairs)\n")

    # ── Per-CV means of the headline 0–100 scores ──────────────────────────
    headline = ["L1_P2_score_100", "L2_fit_score_100",
                "L2_weighted_from_subscales_100", "L3_fit_score_0_100"]
    print("Per-CV mean of headline 0–100 scores:")
    print(df.groupby("cv_id")[headline].mean().round(1).to_string())
    print()

    # ── Per-CV mean by jd_pool, L3 final score ────────────────────────────
    print("L3_fit_score_0_100 mean by (cv_id × jd_pool):")
    print(df.pivot_table(index="cv_id", columns="jd_pool",
                         values="L3_fit_score_0_100", aggfunc="mean")
            .round(1).to_string())
    print()

    # ── Correlations between the headline 0–100 scores (pooled) ───────────
    print("Pearson r between headline 0–100 scores (all 225 pairs where both exist):")
    sub = df[headline].dropna(how="any")
    print(sub.corr(method="pearson").round(3).to_string())
    print(f"   n = {len(sub)} pairs with all four scores present")
    print()

    print("Spearman ρ on the same subset:")
    print(sub.corr(method="spearman").round(3).to_string())


if __name__ == "__main__":
    main()

