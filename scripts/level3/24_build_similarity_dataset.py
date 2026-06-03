#!/usr/bin/env python3
"""
Level 3 v2 — Step 24: Build the per-pair similarity dataset.

For every (CV, JD) pair where both sides have already been:
  1. Segmented and tagged (script 03 / 09),
  2. Normalised into skills + experience labels + degree (script 22),
  3. Embedded as single skills-block and experience-block vectors (script 23),

this script computes three columns:

  - skills_sim     = cosine(cv.skills_emb, jd.skills_emb)            in [-1, 1]
  - experience_sim = cosine(cv.experience_emb, jd.experience_emb)    in [-1, 1]
  - education_match = 1 / 0   (discrete check, see rules below)

NO weighted aggregation. NO normalisation. Just the raw three numbers per pair
so the user can inspect calibration before deciding on a formula.

Education match rules (matches the user's spec):
  - JD has no education requirement (required_degree == "none")     → match = 1
  - CV's highest_degree is at LEAST as high as JD's required_degree  → match = 1
  - Otherwise (CV strictly below requirement)                        → match = 0

Degree hierarchy (mba treated as same level as master):
    none < high_school < associate < bachelor < (master == mba) < phd

Input:
  - results/level3/segments_normalisation/cv_norm/{cv_id}.json
  - results/level3/segments_normalisation/cv_norm/{cv_id}_embeddings.npz
  - results/level3/segments_normalisation/jd_norm/{jd_id}.json
  - results/level3/segments_normalisation/jd_norm/{jd_id}_embeddings.npz

Output:
  - results/level3/segments_normalisation/pair_similarities.csv

Run:
    python3 scripts/level3/24_build_similarity_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT     = Path(__file__).resolve().parent.parent.parent
BASE     = ROOT / "results" / "level3" / "segments_normalisation"
CV_DIR   = BASE / "cv_norm"
JD_DIR   = BASE / "jd_norm"
OUT_CSV  = BASE / "pair_similarities.csv"


# Lowest → highest. MBA shares rank with Master because they are the same
# academic level (and treating them separately would cause MBA candidates
# applying to Master-required roles to fail the check unfairly).
EDU_RANK = {
    "none":        0,
    "high_school": 1,
    "associate":   2,
    "bachelor":    3,
    "master":      4,
    "mba":         4,
    "phd":         5,
}


def list_doc_ids(folder: Path) -> list[str]:
    """All <doc_id>.json files in folder (excluding the *_embeddings.npz)."""
    return sorted(p.stem for p in folder.glob("*.json")
                  if not p.stem.endswith("_embeddings"))


def load_doc(doc_id: str, folder: Path) -> dict | None:
    """Load the normalisation JSON + the embeddings .npz for one document.
    Returns None if either is missing."""
    json_path = folder / f"{doc_id}.json"
    emb_path  = folder / f"{doc_id}_embeddings.npz"
    if not json_path.exists() or not emb_path.exists():
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    emb     = np.load(emb_path, allow_pickle=True)
    return {
        "doc_id":         doc_id,
        "skills_emb":     emb["skills_emb"],
        "experience_emb": emb["experience_emb"],
        "n_skills":       int(emb["n_skills"].item()),
        "n_experience":   int(emb["n_experience"].item()),
        "skills_text":    str(emb["skills_text"].item()),
        "experience_text":str(emb["experience_text"].item()),
        "education":      payload.get("education", {}) or {},
    }


def education_match(cv_edu: dict, jd_edu: dict) -> int:
    """Discrete 0/1 check. JD's required_degree drives the comparison;
    CV's highest_degree is what's compared against it."""
    jd_req = (jd_edu.get("required_degree") or "none").lower()
    cv_hi  = (cv_edu.get("highest_degree")  or "none").lower()
    if jd_req == "none":          # JD has no requirement
        return 1
    return 1 if EDU_RANK.get(cv_hi, 0) >= EDU_RANK.get(jd_req, 0) else 0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """L2-normalised dot product. If either side is a zero vector
    (no labels were produced), returns NaN — caller can treat as missing."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv", default=None,
                    help="Single CV id. Omit to process all available CVs.")
    ap.add_argument("--jd", default=None,
                    help="Single JD id. Omit to process all available JDs.")
    ap.add_argument("--out", default=str(OUT_CSV),
                    help=f"Output CSV path (default: {OUT_CSV.relative_to(ROOT)})")
    args = ap.parse_args()

    cv_ids = [args.cv] if args.cv else list_doc_ids(CV_DIR)
    jd_ids = [args.jd] if args.jd else list_doc_ids(JD_DIR)

    # Pre-load CV side once (small set).
    cv_data: dict[str, dict] = {}
    for cid in cv_ids:
        d = load_doc(cid, CV_DIR)
        if d is None:
            print(f"  ⚠  CV '{cid}' missing JSON or .npz — skipped")
            continue
        cv_data[cid] = d

    # Process JD side and emit rows.
    rows: list[dict] = []
    for jid in jd_ids:
        jd = load_doc(jid, JD_DIR)
        if jd is None:
            print(f"  ⚠  JD '{jid}' missing JSON or .npz — skipped")
            continue
        for cid, cv in cv_data.items():
            row = {
                "cv_id":          cid,
                "jd_id":          jid,
                "skills_sim":     round(cosine(cv["skills_emb"],     jd["skills_emb"]),     4),
                "experience_sim": round(cosine(cv["experience_emb"], jd["experience_emb"]), 4),
                "education_match": education_match(cv["education"], jd["education"]),
                "cv_highest_degree":  cv["education"].get("highest_degree", "none"),
                "jd_required_degree": jd["education"].get("required_degree", "none"),
                "n_skills_cv":     cv["n_skills"],
                "n_experience_cv": cv["n_experience"],
                "n_skills_jd":     jd["n_skills"],
                "n_experience_jd": jd["n_experience"],
            }
            rows.append(row)

    if not rows:
        print("No pairs produced.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"✓ {len(rows)} pairs  →  {out_path.relative_to(ROOT)}\n")

    # Quick console summary
    print(f"{'cv_id':<14} {'jd_id':<55} {'sk_sim':>7} {'ex_sim':>7} {'edu':>4}")
    print("─" * 92)
    for r in rows:
        print(f"{r['cv_id']:<14} {r['jd_id'][:54]:<55} "
              f"{r['skills_sim']:>7.3f} {r['experience_sim']:>7.3f} {r['education_match']:>4}")


if __name__ == "__main__":
    main()
