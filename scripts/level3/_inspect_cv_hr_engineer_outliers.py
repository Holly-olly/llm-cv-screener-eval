#!/usr/bin/env python3
"""Ad-hoc diagnostic: why does cv_hr score so high on a few engineer JDs?

For each of the named pairs, dump:
  1) headline scores from the merged table (raw + linear-adj + final fit)
  2) tag composition on both sides
  3) per-construct contribution table:
       JD-segment text  →  best CV-segment text  →  cosine  →  CV tag
     This is exactly what the MaxSim block aggregator averages — so it's
     the right place to look for which segments are inflating the score.
  4) top-N highest cells globally (across the whole matrix), with texts
     and tags on both sides.

Read-only. No new files written.

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/_inspect_cv_hr_engineer_outliers.py
"""
from pathlib import Path
import numpy as np
import pandas as pd


ROOT     = Path(__file__).parent.parent.parent
MAT_DIR  = ROOT / "results" / "level3" / "similarity_matrices"
ALL_CSV  = ROOT / "results" / "level3" / "all_levels_comparison.csv"

CVS = ["cv_primary", "cv_hr", "cv_engineer"]
JDS = [
    "zs_business_technology_solutions_associate_consultant_5a0be3",
    "meta_software_engineer_leadership_machine_learning_8bf9b1",
    "Klaviyo_Senior_Solution_Architect",
]

TOP_N_CELLS_GLOBAL = 15
SHOW_TEXT_CHARS    = 110


def short(t: str) -> str:
    t = " ".join(str(t).split())
    return t if len(t) <= SHOW_TEXT_CHARS else t[:SHOW_TEXT_CHARS - 1] + "…"


def construct_contribution(sim, cv_tags, cv_texts, jd_tags, jd_texts, construct: str):
    """For each JD segment that contributes to `construct`, find its best CV
    match (restricted to allowed CV tags). Returns a sorted DataFrame.
    """
    allowed_cv = np.isin(cv_tags, [construct, "mixed"]) if construct != "education" \
                 else (cv_tags == "education")
    jd_mask = (jd_tags == construct)
    if construct in ("skills", "experience"):
        jd_mask = jd_mask | (jd_tags == "mixed")
    if not allowed_cv.any() or not jd_mask.any():
        return pd.DataFrame()

    sub_sim   = sim[allowed_cv, :][:, jd_mask]                # (n_cv_allowed, n_jd_kept)
    cv_sub_ix = np.where(allowed_cv)[0]
    jd_sub_ix = np.where(jd_mask)[0]

    rows = []
    for j_local, j_global in enumerate(jd_sub_ix):
        col = sub_sim[:, j_local]
        best_local = int(col.argmax())
        i_global   = int(cv_sub_ix[best_local])
        rows.append({
            "jd_tag":  str(jd_tags[j_global]),
            "jd_text": short(jd_texts[j_global]),
            "cv_tag":  str(cv_tags[i_global]),
            "cv_text": short(cv_texts[i_global]),
            "cosine":  float(col[best_local]),
            "weight":  1.0 if jd_tags[j_global] == construct else 0.5,
        })
    out = pd.DataFrame(rows).sort_values("cosine", ascending=False).reset_index(drop=True)
    return out


def top_cells_global(sim, cv_tags, cv_texts, jd_tags, jd_texts, n: int):
    """Top-N highest-cosine cells across the whole matrix (ignoring nothing —
    `other` segments were already excluded when the npz was built)."""
    flat = sim.flatten()
    idx  = np.argpartition(-flat, min(n, flat.size - 1))[:n]
    idx  = idx[np.argsort(-flat[idx])]
    rows = []
    n_jd = sim.shape[1]
    for k in idx:
        i, j = int(k // n_jd), int(k % n_jd)
        rows.append({
            "cosine":  float(sim[i, j]),
            "cv_tag":  str(cv_tags[i]),
            "cv_text": short(cv_texts[i]),
            "jd_tag":  str(jd_tags[j]),
            "jd_text": short(jd_texts[j]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.max_colwidth", SHOW_TEXT_CHARS + 5)
    pd.set_option("display.width", 250)

    full = pd.read_csv(ALL_CSV)

    # ── Compact comparison table first — easy CV-vs-CV scan on the same JDs ─
    headline_cols = ["cv_id", "jd_id", "jd_pool",
                     "L2_holistic", "L2_fit_score_100",
                     "L3_skills_block", "L3_experience_block", "L3_education_block",
                     "L3_skills_norm", "L3_experience_norm", "L3_education_norm",
                     "L3_fit_score_0_100",
                     "L3_edu_required", "L3_edu_fallback_used"]
    side_by_side = (full[full["cv_id"].isin(CVS) & full["jd_id"].isin(JDS)]
                    [headline_cols]
                    .sort_values(["jd_id", "cv_id"])
                    .reset_index(drop=True))
    print("══ HEADLINE — 3 CVs × 3 JDs side-by-side ══")
    print(side_by_side.to_string(index=False))

    for CV in CVS:
      headline = full[full["cv_id"] == CV].set_index("jd_id")

      for jd in JDS:
        npz_path = MAT_DIR / f"{CV}__vs__{jd}.npz"
        print(f"\n{'═' * 100}")
        print(f"  {CV}  ×  {jd}")
        print('═' * 100)

        if jd not in headline.index:
            print(f"  (no row in all_levels_comparison.csv for this pair)")
        else:
            h = headline.loc[jd]
            print("Headline scores")
            print(f"  jd_pool                    : {h['jd_pool']}")
            print(f"  L2_holistic / score        : {h['L2_holistic']} / {h['L2_holistic_score']}")
            print(f"  L2_fit_score_100           : {h['L2_fit_score_100']}")
            print(f"  L3 raw blocks  (sk/ex/ed)  : "
                  f"{h['L3_skills_block']:.4f} / {h['L3_experience_block']:.4f} / {h['L3_education_block']:.4f}")
            print(f"  L3 linear-adj  (sk/ex/ed)  : "
                  f"{h['L3_skills_norm']:.4f} / {h['L3_experience_norm']:.4f} / {h['L3_education_norm']:.4f}")
            print(f"  L3_fit_score_0_100         : {h['L3_fit_score_0_100']}")
            print(f"  L3_edu_required / fallback : {h['L3_edu_required']} / {h['L3_edu_fallback_used']}")

        if not npz_path.exists():
            print(f"  (matrix file missing: {npz_path.name})")
            continue

        data = np.load(npz_path, allow_pickle=True)
        sim      = data["sim"]
        cv_tags  = data["cv_tags"]
        cv_texts = data["cv_texts"]
        jd_tags  = data["jd_tags"]
        jd_texts = data["jd_texts"]

        print(f"\nTag composition")
        print(f"  CV non-other segments: n={sim.shape[0]}  "
              f"tags={dict(pd.Series(cv_tags).value_counts())}")
        print(f"  JD non-other segments: n={sim.shape[1]}  "
              f"tags={dict(pd.Series(jd_tags).value_counts())}")

        for construct in ("skills", "experience", "education"):
            print(f"\n── {construct.upper()} block — per-JD-segment best-match ─────")
            df = construct_contribution(sim, cv_tags, cv_texts, jd_tags, jd_texts, construct)
            if df.empty:
                print("   (empty — either side has no qualifying segment)")
                continue
            # Weighted mean = the MaxSim block aggregator's exact output.
            wmean = (df["cosine"] * df["weight"]).sum() / df["weight"].sum()
            print(f"   weighted-mean cosine (block_{construct}) = {wmean:.4f}   "
                  f"(n_jd_segments contributing = {len(df)})")
            print(df.to_string(index=False))

        print(f"\n── TOP-{TOP_N_CELLS_GLOBAL} cells across the whole matrix ─────")
        tc = top_cells_global(sim, cv_tags, cv_texts, jd_tags, jd_texts, TOP_N_CELLS_GLOBAL)
        print(tc.to_string(index=False))


if __name__ == "__main__":
    main()
