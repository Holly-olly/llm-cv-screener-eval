#!/usr/bin/env python3
"""
Level 3 v2 — Step 23: Embed normalised skills + experience blocks.

For each CV and JD that already went through script 22
(`22_llm_normalize_skills.py`), this script:

  1. Reads the normalised annotations JSON.
  2. Collects every annotation with tag = 'skill' into ONE concatenated
     "skills text" (comma-joined, case-insensitively de-duplicated).
  3. Does the same for 'experience'.
  4. Embeds each block as a single 384-d MiniLM vector
     (L2-normalised, so cosine similarity == dot product).
  5. Writes `<doc_id>_embeddings.npz` next to the source JSON.

Education is NOT embedded — it is matched via the discrete `degree`
field already stored in the JSON (see script 22).

Input:
  - results/level3/segments_normalisation/cv_norm/<cv_id>.json
  - results/level3/segments_normalisation/jd_norm/<jd_id>.json

Output (per document):
  - <same folder>/<doc_id>_embeddings.npz with keys:
      doc_id, doc_type, model,
      skills_text, skills_labels, skills_emb,
      experience_text, experience_labels, experience_emb,
      n_skills, n_experience

Run:
    # All CVs + all JDs already normalised
    python3 scripts/level3/23_embed_normalized.py

    # One document only
    python3 scripts/level3/23_embed_normalized.py --cv cv_primary
    python3 scripts/level3/23_embed_normalized.py --jd codility_assessment_scientist
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT     = Path(__file__).resolve().parent.parent.parent
BASE     = ROOT / "results" / "level3" / "segments_normalisation"
CV_DIR   = BASE / "cv_norm"
JD_DIR   = BASE / "jd_norm"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ──────────────────────────────────────────────────────────────────────────
# Per-document block extraction
# ──────────────────────────────────────────────────────────────────────────
def _dedupe_preserve_order(labels: list[str]) -> list[str]:
    """Case-insensitive de-dupe that keeps the first occurrence's casing."""
    seen: set[str] = set()
    out: list[str] = []
    for lab in labels:
        key = lab.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(lab.strip())
    return out


def extract_blocks(payload: dict) -> tuple[list[str], list[str]]:
    """Return (skills_labels, experience_labels) — de-duped, in the order
    the LLM annotations were produced."""
    skills: list[str] = []
    exps: list[str] = []
    for item in payload.get("items", []):
        for ann in item.get("annotations", []):
            label = (ann.get("normalized_label") or "").strip()
            if not label:
                continue
            if ann.get("tag") == "skill":
                skills.append(label)
            elif ann.get("tag") == "experience":
                exps.append(label)
    return _dedupe_preserve_order(skills), _dedupe_preserve_order(exps)


def concat_text(labels: list[str]) -> str:
    """Single concatenated string suitable for sentence-transformer encoding."""
    return ", ".join(labels)


# ──────────────────────────────────────────────────────────────────────────
# Embedding
# ──────────────────────────────────────────────────────────────────────────
def embed_blocks(model: SentenceTransformer, labels: list[str]) -> tuple[str, np.ndarray]:
    """Concat → encode → return (text, L2-normalised vector). Empty block
    returns ('', zero-vector of the model's embed dim)."""
    text = concat_text(labels)
    if not text:
        return "", np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
    vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return text, vec.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# Per-document driver
# ──────────────────────────────────────────────────────────────────────────
def process_one(model: SentenceTransformer, src_path: Path,
                doc_type: str, force: bool) -> dict | None:
    out_path = src_path.with_name(f"{src_path.stem}_embeddings.npz")
    if out_path.exists() and not force:
        return {"doc_id": src_path.stem, "doc_type": doc_type,
                "skipped": True, "reason": "already exists"}

    if not src_path.exists():
        return {"doc_id": src_path.stem, "doc_type": doc_type,
                "skipped": True, "reason": "source JSON missing"}

    payload = json.loads(src_path.read_text(encoding="utf-8"))
    skills_labels, exp_labels = extract_blocks(payload)
    skills_text, skills_vec   = embed_blocks(model, skills_labels)
    exp_text,    exp_vec      = embed_blocks(model, exp_labels)

    np.savez(
        out_path,
        doc_id          = np.array([src_path.stem]),
        doc_type        = np.array([doc_type]),
        model           = np.array([str(model)]),
        skills_text     = np.array([skills_text]),
        skills_labels   = np.array(skills_labels) if skills_labels else np.array([], dtype=object),
        skills_emb      = skills_vec,
        experience_text = np.array([exp_text]),
        experience_labels = np.array(exp_labels) if exp_labels else np.array([], dtype=object),
        experience_emb  = exp_vec,
        n_skills        = np.array([len(skills_labels)]),
        n_experience    = np.array([len(exp_labels)]),
    )
    return {"doc_id": src_path.stem, "doc_type": doc_type,
            "n_skills": len(skills_labels), "n_experience": len(exp_labels),
            "skills_zero":     bool(np.allclose(skills_vec, 0)),
            "experience_zero": bool(np.allclose(exp_vec, 0)),
            "skipped": False}


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv", default=None,
                   help="Single CV id (file: cv_norm/<id>.json). Omit to process all.")
    p.add_argument("--jd", default=None,
                   help="Single JD id (file: jd_norm/<id>.json). Omit to process all.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="sentence-transformer model name (default: MiniLM-L6-v2)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing *_embeddings.npz files.")
    args = p.parse_args()

    # Resolve which docs to process
    if args.cv or args.jd:
        cv_paths = [CV_DIR / f"{args.cv}.json"] if args.cv else []
        jd_paths = [JD_DIR / f"{args.jd}.json"] if args.jd else []
    else:
        cv_paths = sorted(p for p in CV_DIR.glob("*.json")
                          if not p.stem.endswith("_embeddings"))
        jd_paths = sorted(p for p in JD_DIR.glob("*.json")
                          if not p.stem.endswith("_embeddings"))

    print(f"Scope: {len(cv_paths)} CV(s) + {len(jd_paths)} JD(s) "
          f"= {len(cv_paths) + len(jd_paths)} documents")
    print(f"Model: {args.model}    force={args.force}")
    print()

    print(f"Loading model {args.model} …")
    model = SentenceTransformer(args.model)
    print(f"  embedding dim = {model.get_sentence_embedding_dimension()}\n")

    print(f"{'doc_id':<55} {'type':<3} {'n_sk':>4} {'n_exp':>5}  status")
    print("─" * 85)
    for path in cv_paths:
        row = process_one(model, path, "cv", args.force)
        print_row(row)
    for path in jd_paths:
        row = process_one(model, path, "jd", args.force)
        print_row(row)


def print_row(row: dict | None) -> None:
    if not row:
        return
    if row.get("skipped"):
        print(f"{row['doc_id']:<55} {row['doc_type']:<3} {'-':>4} {'-':>5}  "
              f"skip ({row['reason']})")
    else:
        flags = []
        if row.get("skills_zero"):     flags.append("skills_empty")
        if row.get("experience_zero"): flags.append("exp_empty")
        suffix = "  " + ",".join(flags) if flags else ""
        print(f"{row['doc_id']:<55} {row['doc_type']:<3} "
              f"{row['n_skills']:>4} {row['n_experience']:>5}  written{suffix}")


if __name__ == "__main__":
    main()
