#!/usr/bin/env python3
"""
Level 3 — Step 9: Label CV lines using the SAME taxonomy and prompt as
Stage 1 used for JDs.

Same five tags (skills / experience / education / mixed / other), same
few-shot examples (from `data/labeled_rag_jd/`), same generation settings,
same JSON I/O schema, same retry-with-backoff and autofill safety nets.
The only change is the input source (CV .txt files instead of JD .txt
files) and the output directory.

Reads:
  - data/cv/*.txt

Writes per CV:
  - results/level3/llm_labelled_json_cv/{cv_id}_input.json
  - results/level3/llm_labelled_json_cv/{cv_id}_output.json
  - results/level3/llm_labelled_json_cv/{cv_id}.csv     (joined view)

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/09_label_cvs.py
"""

import importlib.util
import json
from pathlib import Path

from openai import OpenAI


ROOT    = Path(__file__).parent.parent.parent
CV_DIR  = ROOT / "data" / "cv"
OUT_DIR = ROOT / "results" / "level3" / "llm_labelled_json_cv"

# ── Reuse logic from 03_llm_label_json.py (leading digit forces importlib) ─
_SPEC = importlib.util.spec_from_file_location(
    "label_json", Path(__file__).parent / "03_llm_label_json.py"
)
_M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_M)

SYSTEM_PROMPT          = _M.SYSTEM_PROMPT
build_user_prompt      = _M.build_user_prompt
strip_blanks_to_json   = _M.strip_blanks_to_json
load_openai_key        = _M.load_openai_key
load_examples          = _M.load_examples
call_openai_with_retry = _M.call_openai_with_retry
autofill_missing       = _M.autofill_missing
validate               = _M.validate
ALLOWED_TAGS           = _M.ALLOWED_TAGS
SKIP_EXISTING          = True


def write_outputs(cv_id: str, target_input: list[dict],
                  output: dict, auto_filled: set[int]) -> tuple[Path, Path, Path]:
    import csv as _csv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    in_path  = OUT_DIR / f"{cv_id}_input.json"
    out_path = OUT_DIR / f"{cv_id}_output.json"
    csv_path = OUT_DIR / f"{cv_id}.csv"

    in_path.write_text(json.dumps(target_input, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.write_text(json.dumps(output,       ensure_ascii=False, indent=2), encoding="utf-8")

    text_by_id = {row["line_id"]: row["text"] for row in target_input}
    tag_by_id  = {row["line_id"]: row.get("tag") for row in output.get("labels", [])}
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["line_id", "tag", "auto_filled", "text"])
        for lid in sorted(text_by_id):
            w.writerow([lid, tag_by_id.get(lid, ""),
                        "Y" if lid in auto_filled else "",
                        text_by_id[lid]])
    return in_path, out_path, csv_path


def main() -> None:
    cv_files = sorted(CV_DIR.glob("*.txt"))
    if not cv_files:
        raise SystemExit(f"No CV files in {CV_DIR}")

    examples = load_examples()
    if not examples:
        raise SystemExit("No few-shot examples available.")

    client = OpenAI(api_key=load_openai_key())

    print(f"Few-shot examples reused from labeled_rag_jd/: {len(examples)} JDs")
    print(f"CVs to label: {len(cv_files)}\n")

    rows = []
    n_ok = n_skip = 0
    total_usage = {"prompt": 0, "completion": 0, "total": 0}

    for cv_path in cv_files:
        cv_id = cv_path.stem
        out_json = OUT_DIR / f"{cv_id}_output.json"
        if SKIP_EXISTING and out_json.exists():
            print(f"── {cv_id} ──  (skip — output exists)")
            n_skip += 1
            continue

        raw = cv_path.read_text(encoding="utf-8")
        target_input = strip_blanks_to_json(raw)

        print(f"── {cv_id} ──  ({len(raw.splitlines())} raw → "
              f"{len(target_input)} content lines)")
        user_prompt = build_user_prompt(examples, target_input)

        try:
            data, usage = call_openai_with_retry(client, SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            print(f"  ✗ {e}\n")
            continue

        for k in total_usage:
            total_usage[k] += usage[k]

        auto_filled = autofill_missing(target_input, data)
        v = validate(target_input, data)
        if v["all_ok"]:
            n_ok += 1
        in_path, out_path, csv_path = write_outputs(cv_id, target_input, data, auto_filled)

        status = "✓" if v["all_ok"] else "✗"
        print(f"  {status} input={v['n_input']} → output={v['n_output']}  order_ok={v['order_ok']}")
        if auto_filled:
            print(f"    ⓘ auto-filled {len(auto_filled)} missing line_ids with tag=other")
        if v["missing_ids"]:
            print(f"    ! missing line_ids: {v['missing_ids'][:10]}")
        if v["bad_tag_lines"]:
            print(f"    ! {len(v['bad_tag_lines'])} bad-tag lines")
            for lid, t in v["bad_tag_lines"][:3]:
                print(f"        line_id {lid}: {t}")
        print(f"    tags: {v['tag_distribution']}")
        print(f"    tokens: {usage['prompt']:,} + {usage['completion']:,} = {usage['total']:,}")
        print(f"    csv:    {csv_path.relative_to(ROOT)}\n")

        row = {"cv_id": cv_id, "lines": v["n_input"]}
        row.update({t: v["tag_distribution"].get(t, 0) for t in ALLOWED_TAGS})
        rows.append(row)

    # ── Cross-CV table ──────────────────────────────────────────────────────
    if rows:
        print("Tag distribution across CVs:")
        cols = ["cv_id", "lines"] + ALLOWED_TAGS
        widths = {c: max(len(c), max((len(str(r[c])) for r in rows), default=0))
                  for c in cols}
        widths["cv_id"] = min(widths["cv_id"], 30)
        print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
        print("  " + "  ".join("-" * widths[c] for c in cols))
        for r in rows:
            print("  " + "  ".join(str(r[c])[:widths[c]].ljust(widths[c]) for c in cols))

    n_processed = len(cv_files) - n_skip
    print(f"\nRun summary: {n_ok}/{n_processed} validated cleanly  (skipped {n_skip})")
    if n_processed > 0:
        cost = (total_usage['prompt'] * 0.150 + total_usage['completion'] * 0.600) / 1_000_000
        print(f"Tokens: {total_usage['total']:,}    cost ~${cost:.3f} on gpt-4o-mini")
    print(f"\nReview outputs in: {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
