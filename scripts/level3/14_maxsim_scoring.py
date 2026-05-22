#!/usr/bin/env python3
"""
Level 3 — Step 14: Per-construct MaxSim aggregation.

For every (CV, JD) pair, collapse the pairwise cosine matrix into ONE value
per construct (skills, experience, education). No weighted formula here —
the L2-style 0.6 / 0.3 / 0.1 composite is built later in script 20 from the
linear-normalised per-construct values.

Block definition (asymmetric, MaxSim style):

  for each JD segment j contributing to construct C:
      best[j] = max over CV segments i where cv_tag[i] ∈ allowed(C) of sim[i, j]

  where:
      allowed(skills)     = {skills, mixed}
      allowed(experience) = {experience, mixed}
      allowed(education)  = {education}

  per-segment weight when aggregating block C:
      w_j = 1.0  if jd_tag[j] == C
      w_j = 0.5  if jd_tag[j] == "mixed"   (only for C ∈ {skills, experience})

  block_C = (Σ w_j · best[j]) / (Σ w_j)

Education fallback rule:
  - If JD has 0 education chunks → education_block = mean(skills_block,
    experience_block); edu_fallback_used = True.
  - If JD requires education AND CV has at least one education chunk →
    MaxSim on (CV edu × JD edu).
  - If JD requires education AND CV has none → NaN (caller decides
    how to propagate downstream).

Reads:
  - results/level3/similarity_matrices/*.npz
  - results/level3/labelled/jd/{jd_id}.json   (for pool lookup)

Writes:
  - results/level3/maxsim_summary.csv

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/14_maxsim_scoring.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT     = Path(__file__).parent.parent.parent
MAT_DIR  = ROOT / "results" / "level3" / "similarity_matrices"
JD_DIR   = ROOT / "results" / "level3" / "labelled" / "jd"
OUT_CSV  = ROOT / "results" / "level3" / "maxsim_summary.csv"

MIXED_WEIGHT = 0.5
PURE_WEIGHT  = 1.0


def jd_pool_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in JD_DIR.glob("*.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[rec["id"]] = rec.get("pool") or ""
    return out


def construct_block(sim: np.ndarray,
                    cv_tags: np.ndarray,
                    jd_tags: np.ndarray,
                    construct: str) -> float:
    """MaxSim block score with `mixed` JD segments contributing weight 0.5."""
    allowed_cv = np.isin(cv_tags, [construct, "mixed"])
    if not allowed_cv.any():
        return float("nan")
    sub_sim = sim[allowed_cv, :]

    pure_mask  = jd_tags == construct
    mixed_mask = jd_tags == "mixed"
    if not (pure_mask.any() or mixed_mask.any()):
        return float("nan")

    weighted_sum = 0.0
    total_weight = 0.0
    if pure_mask.any():
        best_pure = sub_sim[:, pure_mask].max(axis=0)
        weighted_sum += float(best_pure.sum()) * PURE_WEIGHT
        total_weight += float(pure_mask.sum()) * PURE_WEIGHT
    if mixed_mask.any():
        best_mixed = sub_sim[:, mixed_mask].max(axis=0)
        weighted_sum += float(best_mixed.sum()) * MIXED_WEIGHT
        total_weight += float(mixed_mask.sum()) * MIXED_WEIGHT
    if total_weight == 0.0:
        return float("nan")
    return weighted_sum / total_weight


def education_block(sim: np.ndarray,
                    cv_tags: np.ndarray,
                    jd_tags: np.ndarray) -> float:
    """MaxSim restricted to (CV edu × JD edu). NaN if either side empty."""
    jd_edu = jd_tags == "education"
    cv_edu = cv_tags == "education"
    if not jd_edu.any() or not cv_edu.any():
        return float("nan")
    block = sim[np.ix_(cv_edu, jd_edu)]
    return float(block.max(axis=0).mean()) if block.size else float("nan")


def summarise_one(npz_path: Path, pools: dict[str, str]) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    sim     = data["sim"]
    cv_id   = str(data["cv_id"][0])
    jd_id   = str(data["jd_id"][0])
    cv_tags = data["cv_tags"]
    jd_tags = data["jd_tags"]

    row: dict = {
        "cv_id":   cv_id,
        "jd_id":   jd_id,
        "jd_pool": pools.get(jd_id, ""),
        "n_cv":    int(sim.shape[0]),
        "n_jd":    int(sim.shape[1]),
    }
    for t in ("skills", "experience", "education", "mixed"):
        row[f"n_cv_{t}"] = int((cv_tags == t).sum())
        row[f"n_jd_{t}"] = int((jd_tags == t).sum())

    # Per-construct MaxSim blocks
    sk = construct_block(sim, cv_tags, jd_tags, "skills")
    ex = construct_block(sim, cv_tags, jd_tags, "experience")
    row["skills_block"]     = round(sk, 4) if not np.isnan(sk) else float("nan")
    row["experience_block"] = round(ex, 4) if not np.isnan(ex) else float("nan")

    # Education block with the fallback rule
    jd_edu_mask = jd_tags == "education"
    cv_edu_mask = cv_tags == "education"
    row["edu_required"]      = bool(jd_edu_mask.any())
    row["edu_fallback_used"] = False

    if not jd_edu_mask.any():
        valid = [v for v in (sk, ex) if not np.isnan(v)]
        edu_val = float(np.mean(valid)) if valid else float("nan")
        row["edu_fallback_used"] = True
    elif cv_edu_mask.any():
        edu_val = education_block(sim, cv_tags, jd_tags)
    else:
        edu_val = float("nan")

    row["education_block"] = round(edu_val, 4) if not np.isnan(edu_val) else float("nan")
    return row


def main() -> None:
    pools = jd_pool_map()
    files = sorted(MAT_DIR.glob("*.npz"))
    if not files:
        raise SystemExit(f"No matrices in {MAT_DIR}")

    rows = [summarise_one(p, pools) for p in files]
    df = pd.DataFrame(rows)

    col_order = [
        "cv_id", "jd_id", "jd_pool",
        "n_cv", "n_jd",
        "n_cv_skills", "n_cv_experience", "n_cv_education", "n_cv_mixed",
        "n_jd_skills", "n_jd_experience", "n_jd_education", "n_jd_mixed",
        "skills_block", "experience_block",
        "education_block", "edu_required", "edu_fallback_used",
    ]
    df = df[col_order].sort_values(["cv_id", "jd_id"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ MaxSim summary → {OUT_CSV.relative_to(ROOT)}  ({len(df)} pairs)\n")

    cols_show = ["jd_pool", "jd_id",
                 "skills_block", "experience_block", "education_block",
                 "edu_required", "edu_fallback_used"]
    for cv in df["cv_id"].unique():
        sub = df[df["cv_id"] == cv].sort_values("skills_block", ascending=False)
        print(f"── {cv} ──  (TOP 5 by skills_block)")
        with pd.option_context("display.max_colwidth", 55, "display.width", 200):
            print(sub[cols_show].head(5).to_string(index=False))
        print()

    print("Per-construct corpus stats:")
    print(df[["skills_block", "experience_block", "education_block"]]
          .describe().round(3).to_string())


if __name__ == "__main__":
    main()
