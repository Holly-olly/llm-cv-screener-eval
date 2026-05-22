#!/usr/bin/env python3
"""
Level 3 — Step 12: Build cosine similarity matrices for every CV × JD pair.

For each combination of one CV and one JD, compute the full pairwise cosine
similarity matrix between every NON-`other` CV segment and every non-`other`
JD segment, and persist it as a `.npz` file containing the matrix plus
per-side metadata (line_id, tag, text).

This script intentionally does NOT compute aggregate scores — it just
materialises the raw similarity matrices. A separate script (13) reads
these matrices and produces summary tables.

Output naming:
  results/level3/similarity_matrices/{cv_id}__vs__{jd_id}.npz

Idempotent: pairs whose .npz already exists are skipped (unless --force).

Reads:
  - results/level3/labelled/cv/{cv_id}.json
  - results/level3/labelled/jd/{jd_id}.json

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/12_build_similarity_matrices.py
    python3 scripts/level3/12_build_similarity_matrices.py --force
"""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT     = Path(__file__).parent.parent.parent
CV_DIR   = ROOT / "results" / "level3" / "labelled" / "cv"
JD_DIR   = ROOT / "results" / "level3" / "labelled" / "jd"
OUT_DIR  = ROOT / "results" / "level3" / "similarity_matrices"

EXCLUDE_TAG = "other"


def load_segments(path: Path) -> tuple[list[dict], np.ndarray]:
    rec = json.loads(path.read_text(encoding="utf-8"))
    rows = [s for s in rec["segments"] if s.get("tag") != EXCLUDE_TAG]
    meta = [{"line_id": s["line_id"], "tag": s["tag"], "text": s["text"]} for s in rows]
    if rows:
        emb = np.array([s["embedding"] for s in rows], dtype=np.float32)
    else:
        emb = np.zeros((0, 384), dtype=np.float32)
    return meta, emb


def save_pair(cv_id: str, jd_id: str, cv_meta: list[dict], jd_meta: list[dict],
              sim: np.ndarray, out_path: Path) -> None:
    np.savez_compressed(
        out_path,
        sim          = sim,
        cv_id        = np.array([cv_id]),
        jd_id        = np.array([jd_id]),
        cv_line_ids  = np.array([m["line_id"] for m in cv_meta]),
        cv_tags      = np.array([m["tag"]     for m in cv_meta]),
        cv_texts     = np.array([m["text"]    for m in cv_meta]),
        jd_line_ids  = np.array([m["line_id"] for m in jd_meta]),
        jd_tags      = np.array([m["tag"]     for m in jd_meta]),
        jd_texts     = np.array([m["text"]    for m in jd_meta]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Recompute matrices even if the .npz already exists.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cv_paths = sorted(CV_DIR.glob("*.json"))
    jd_paths = sorted(JD_DIR.glob("*.json"))
    print(f"CVs: {len(cv_paths)}, JDs: {len(jd_paths)}, "
          f"pairs to consider: {len(cv_paths) * len(jd_paths)}")

    # Pre-load CV embeddings once
    cv_cache: dict[str, tuple[list[dict], np.ndarray]] = {}
    for p in cv_paths:
        cv_cache[p.stem] = load_segments(p)
        meta, _ = cv_cache[p.stem]
        print(f"  cv:  {p.stem:<20}  non-other segments = {len(meta)}")
    print()

    n_done = n_skipped = n_empty = 0
    for jd_path in jd_paths:
        jd_id = jd_path.stem
        jd_meta, jd_emb = load_segments(jd_path)
        if len(jd_meta) == 0:
            n_empty += 1
            print(f"  ⓘ skip {jd_id} — no non-other segments")
            continue

        for cv_id, (cv_meta, cv_emb) in cv_cache.items():
            out_path = OUT_DIR / f"{cv_id}__vs__{jd_id}.npz"
            if out_path.exists() and not args.force:
                n_skipped += 1
                continue
            if len(cv_meta) == 0:
                continue

            sim = (cv_emb @ jd_emb.T).astype(np.float32)
            save_pair(cv_id, jd_id, cv_meta, jd_meta, sim, out_path)
            n_done += 1

    print(f"\n✓ {n_done} matrices built, {n_skipped} skipped (already exist), "
          f"{n_empty} JDs had no non-other segments")
    print(f"  Output dir: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
