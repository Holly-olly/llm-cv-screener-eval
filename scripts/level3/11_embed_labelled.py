#!/usr/bin/env python3
"""
Level 3 — Step 11: Add embeddings to every segment in every labelled
JSON file (JDs and CVs).

For each file in `results/level3/labelled/{jd,cv}/`, this script:
  1. Loads the file.
  2. Embeds every segment's `text` with sentence-transformers
     `all-MiniLM-L6-v2`, L2-normalised, mean-pooled (model default).
  3. Adds an `embedding` field (list of 384 floats) to each segment.
  4. Writes the file back in place (atomic: write to .tmp then rename).

No similarity computation here — that lives in a later stage. The
embeddings are stored as float lists rounded to 5 decimal places to
keep JSON readable without losing meaningful precision (MiniLM components
are bounded in roughly [-0.3, 0.3] before normalisation, and [-1, 1]
after; 5 decimal places give ~10⁻⁵ resolution which is far below the
model's intrinsic noise floor).

Idempotent:
  - If a segment already has an `embedding` and `--force` is not set,
    the file is skipped.

Reads / writes (in place):
  - results/level3/labelled/jd/*.json
  - results/level3/labelled/cv/*.json

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/11_embed_labelled.py
    python3 scripts/level3/11_embed_labelled.py --force   # re-embed everything
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT     = Path(__file__).parent.parent.parent
LBL_DIR  = ROOT / "results" / "level3" / "labelled"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROUND_TO   = 5


def file_needs_embedding(path: Path) -> bool:
    rec = json.loads(path.read_text(encoding="utf-8"))
    if not rec.get("segments"):
        return False
    return any("embedding" not in s for s in rec["segments"])


def embed_file(model: SentenceTransformer, path: Path) -> tuple[int, int]:
    """Embed any segment in `path` that doesn't already have an embedding.
    Returns (n_segments, n_newly_embedded).
    """
    rec = json.loads(path.read_text(encoding="utf-8"))
    segments = rec.get("segments", [])
    if not segments:
        return 0, 0

    todo_idx, todo_texts = [], []
    for i, s in enumerate(segments):
        if "embedding" not in s:
            todo_idx.append(i)
            todo_texts.append(s["text"])

    if not todo_idx:
        return len(segments), 0

    vecs = model.encode(todo_texts,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=64)

    for i, v in zip(todo_idx, vecs):
        segments[i]["embedding"] = [round(float(x), ROUND_TO) for x in v]

    rec["embedding_meta"] = {
        "model":       MODEL_NAME,
        "dim":         int(vecs.shape[1]),
        "normalised":  True,
        "round_to":    ROUND_TO,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return len(segments), len(todo_idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Drop existing embeddings and re-embed every segment.")
    args = ap.parse_args()

    files = sorted(LBL_DIR.rglob("*.json"))
    if not files:
        raise SystemExit(f"No labelled JSON files under {LBL_DIR}")

    if args.force:
        for p in files:
            rec = json.loads(p.read_text(encoding="utf-8"))
            for s in rec.get("segments", []):
                s.pop("embedding", None)
            rec.pop("embedding_meta", None)
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"--force: stripped existing embeddings from {len(files)} files")

    pending = [p for p in files if file_needs_embedding(p)]
    print(f"Files: {len(files)} total, {len(pending)} need embedding\n")

    if not pending:
        print("Nothing to do — all files already embedded.")
        return

    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print()

    total_segs = total_new = 0
    for p in pending:
        n_segs, n_new = embed_file(model, p)
        total_segs += n_segs
        total_new  += n_new
        rel = p.relative_to(ROOT)
        print(f"  ✓ {str(rel):<70}  segments={n_segs:>4}  embedded={n_new}")

    print(f"\nTotal segments embedded this run: {total_new:,}")
    print(f"Files updated:                    {len(pending):,}")


if __name__ == "__main__":
    main()
