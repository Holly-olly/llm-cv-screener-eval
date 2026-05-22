#!/usr/bin/env python3
"""
Level 3 — Step 10: Consolidate input/output pairs into one canonical
labelled JSON file per JD and per CV.

This is the source-of-truth artefact for downstream stages. Manual edits
to the per-segment tags should be done in `*_output.json` (or the joined
`.csv`) under `llm_labelled_json/` and `llm_labelled_json_cv/` — re-running
this script then propagates those edits into the canonical files.

Sources (all three feed into `labelled/jd/` or `labelled/cv/`):
  1. LLM-labelled JDs in `llm_labelled_json/`        (72 JDs)
  2. LLM-labelled CVs in `llm_labelled_json_cv/`      (4 CVs)
  3. Hand-labelled few-shot JDs in `data/labeled_rag_jd/`  (3 JDs;
     parsed straight from their inline `[tag] content` format).

Reads:
  - results/level3/llm_labelled_json/{jd_id}_input.json
  - results/level3/llm_labelled_json/{jd_id}_output.json
  - results/level3/llm_labelled_json_cv/{cv_id}_input.json
  - results/level3/llm_labelled_json_cv/{cv_id}_output.json
  - data/labeled_rag_jd/{jd_id}.txt
  - results/level2_master.csv                         (for JD pool assignment)

Writes:
  - results/level3/labelled/jd/{jd_id}.json
  - results/level3/labelled/cv/{cv_id}.json

Embedding preservation:
  If a canonical file already exists, segments whose `text` matches a
  prior segment (same line_id, same text) keep their `embedding` and
  the file-level `embedding_meta`. Only changed/new lines lose their
  embedding and will be re-embedded by `11_embed_labelled.py`.

Schema (per file):
{
  "id":           "Klaviyo_Senior_Solution_Architect",
  "type":         "jd",                       // or "cv"
  "pool":         "engineer_extra",           // or "main"/"hr_extra" for JDs, null for CVs
  "n_lines":      55,
  "tag_counts":   {"skills":..,"experience":..,"education":..,"mixed":..,"other":..},
  "embedding_meta": {...},                    // copied over if any segment kept its embedding
  "segments": [
    {"line_id": 0, "text": "…", "tag": "skills", "embedding": [...]?},
    ...
  ]
}
"""

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd


ROOT      = Path(__file__).parent.parent.parent
JD_LBL    = ROOT / "results" / "level3" / "llm_labelled_json"
CV_LBL    = ROOT / "results" / "level3" / "llm_labelled_json_cv"
FEW_SHOT  = ROOT / "data" / "labeled_rag_jd"
OUT_DIR   = ROOT / "results" / "level3" / "labelled"
L2_MASTER = ROOT / "results" / "level2_master.csv"

TAGS = ["skills", "experience", "education", "mixed", "other"]


# ── Reuse the inline-tag parser from script 03 ──────────────────────────────
_SPEC = importlib.util.spec_from_file_location(
    "label_json", Path(__file__).parent / "03_llm_label_json.py"
)
_M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_M)
parse_labelled_line = _M.parse_labelled_line


def pool_map() -> dict[str, str]:
    if not L2_MASTER.exists():
        return {}
    df = pd.read_csv(L2_MASTER)
    if "source" not in df.columns:
        return {}
    return dict(df.drop_duplicates("jd_id")[["jd_id", "source"]].values)


# ── Embedding-preservation helpers ──────────────────────────────────────────
def load_existing_embeddings(dst: Path) -> tuple[dict[tuple[int, str], list[float]], dict | None]:
    """Return ({(line_id, text): embedding}, embedding_meta or None) from a prior
    canonical file. Empty dict if file does not exist or has no embeddings.
    """
    if not dst.exists():
        return {}, None
    try:
        rec = json.loads(dst.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, None
    out: dict[tuple[int, str], list[float]] = {}
    for s in rec.get("segments", []):
        if "embedding" in s:
            out[(s["line_id"], s["text"])] = s["embedding"]
    return out, rec.get("embedding_meta")


def attach_existing_embeddings(segments: list[dict], dst: Path) -> bool:
    """Mutate `segments` in place, attaching `embedding` where it can be
    reused from `dst`. Returns True if any embedding was carried over.
    """
    cache, _meta = load_existing_embeddings(dst)
    if not cache:
        return False
    carried = 0
    for s in segments:
        key = (s["line_id"], s["text"])
        if key in cache:
            s["embedding"] = cache[key]
            carried += 1
    return carried > 0


# ── Builders for each source ────────────────────────────────────────────────
def from_llm_pair(in_path: Path, out_path: Path) -> dict:
    target_input = json.loads(in_path.read_text(encoding="utf-8"))
    output       = json.loads(out_path.read_text(encoding="utf-8"))

    text_by_id = {row["line_id"]: row["text"] for row in target_input}
    tag_by_id  = {row["line_id"]: row.get("tag") for row in output.get("labels", [])}

    segments = [
        {"line_id": row["line_id"], "text": row["text"], "tag": tag_by_id.get(row["line_id"], "other")}
        for row in target_input
    ]
    return {"segments": segments}


def from_inline_tagged_txt(path: Path) -> dict:
    """Parse a `[tag] content` text file into the canonical segment list."""
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_labelled_line(line)
        if parsed is None:
            continue
        tag, text = parsed
        segments.append({"line_id": len(segments), "text": text, "tag": tag})
    return {"segments": segments}


# ── Write a single canonical file ───────────────────────────────────────────
def write_record(stem: str, kind: str, pool: str | None,
                 body: dict, dst_dir: Path) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{stem}.json"

    segments = body["segments"]
    # Carry over embeddings from prior canonical file where text didn't change
    _ = attach_existing_embeddings(segments, dst)
    _cache, prior_meta = load_existing_embeddings(dst)

    tag_counts = {t: 0 for t in TAGS}
    for s in segments:
        if s["tag"] in tag_counts:
            tag_counts[s["tag"]] += 1

    record: dict = {
        "id":         stem,
        "type":       kind,
        "pool":       pool,
        "n_lines":    len(segments),
        "tag_counts": tag_counts,
    }
    if prior_meta and any("embedding" in s for s in segments):
        record["embedding_meta"] = prior_meta
    record["segments"] = segments

    dst.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


# ── Drivers per source folder ───────────────────────────────────────────────
def build_llm_dir(src_dir: Path, dst_dir: Path, kind: str,
                  pools: dict[str, str]) -> list[dict]:
    rows = []
    for in_path in sorted(src_dir.glob("*_input.json")):
        stem = in_path.stem[:-len("_input")]
        out_path = src_dir / f"{stem}_output.json"
        if not out_path.exists():
            print(f"  ! missing output for {stem}, skipping")
            continue
        body = from_llm_pair(in_path, out_path)
        rec = write_record(
            stem, kind,
            pools.get(stem) if kind == "jd" else None,
            body, dst_dir,
        )
        rows.append(_row(rec))
    return rows


def build_few_shot_jds(dst_dir: Path, pools: dict[str, str]) -> list[dict]:
    rows = []
    for path in sorted(FEW_SHOT.glob("*.txt")):
        stem = path.stem
        body = from_inline_tagged_txt(path)
        if not body["segments"]:
            print(f"  ! no parseable lines in {path.name}")
            continue
        rec = write_record(stem, "jd", pools.get(stem), body, dst_dir)
        rows.append(_row(rec))
    return rows


def _row(rec: dict) -> dict:
    r = {"id": rec["id"], "type": rec["type"], "pool": rec["pool"], "n_lines": rec["n_lines"]}
    r.update(rec["tag_counts"])
    return r


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    pools = pool_map()

    print("── Building JD files (LLM-labelled) ──")
    jd_rows = build_llm_dir(JD_LBL, OUT_DIR / "jd", "jd", pools)
    print(f"  → {len(jd_rows)} files")

    print("\n── Adding JD files (hand-labelled few-shot examples) ──")
    fs_rows = build_few_shot_jds(OUT_DIR / "jd", pools)
    for r in fs_rows:
        pool = r["pool"] or "n/a"
        print(f"  · {r['id']:<45} pool={pool:<15} lines={r['n_lines']:>3}  "
              f"skills={r['skills']:>2} exp={r['experience']:>2} "
              f"edu={r['education']:>2} mixed={r['mixed']:>2} other={r['other']:>2}")
    print(f"  → {len(fs_rows)} few-shot JD files added")

    print("\n── Building CV files ──")
    cv_rows = build_llm_dir(CV_LBL, OUT_DIR / "cv", "cv", pools)
    print(f"  → {len(cv_rows)} files")

    print("\nCV tag summary:")
    for r in cv_rows:
        print(f"  {r['id']:<20} lines={r['n_lines']:>3}  "
              f"skills={r['skills']:>2} exp={r['experience']:>2} "
              f"edu={r['education']:>2} mixed={r['mixed']:>2} other={r['other']:>2}")

    total = len(jd_rows) + len(fs_rows) + len(cv_rows)
    print(f"\nTotal canonical labelled files: {total}  "
          f"(JD: {len(jd_rows) + len(fs_rows)}, CV: {len(cv_rows)})")
    print("Embeddings preserved for segments whose text was unchanged.")
    print("Run scripts/level3/11_embed_labelled.py to (re)embed any new / changed lines.")


if __name__ == "__main__":
    main()
