#!/usr/bin/env python3
"""
Level 3 — Step 4: Descriptive statistics + structural validation for the
LLM-tagged JD corpus.

Reads:
  - results/level3/llm_labelled_json/{jd_id}_input.json
  - results/level3/llm_labelled_json/{jd_id}_output.json

For every JD it computes:
  - n_input_lines, n_output_lines, delta (must be 0 after auto-fill)
  - mean / median / max line length in characters
  - count of each tag: skills, experience, education, mixed, other
  - share (%) of each tag
  - n_auto_filled (lines the LLM forgot; back-filled with tag=other)
  - JD pool (main / hr_extra / engineer_extra) from results/level2_master.csv

Writes:
  - results/level3/validation_stats.csv       per-JD table
  - results/level3/validation_summary.csv     cross-corpus descriptives

Prints:
  - per-JD table (sorted by pool then jd_id)
  - cross-corpus summary (mean / median / sd / min / max per metric)
  - global tag distribution

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/04_validation_stats.py
"""

import json
from pathlib import Path

import pandas as pd


ROOT       = Path(__file__).parent.parent.parent
LBL_DIR    = ROOT / "results" / "level3" / "llm_labelled_json"
L2_MASTER  = ROOT / "results" / "level2_master.csv"
OUT_PER_JD = ROOT / "results" / "level3" / "validation_stats.csv"
OUT_SUM    = ROOT / "results" / "level3" / "validation_summary.csv"

TAGS = ["skills", "experience", "education", "mixed", "other"]


def pool_map() -> dict[str, str]:
    """jd_id → source pool name from the L2 master CSV."""
    if not L2_MASTER.exists():
        return {}
    df = pd.read_csv(L2_MASTER)
    if "source" not in df.columns:
        return {}
    return dict(df.drop_duplicates("jd_id")[["jd_id", "source"]].values)


def main() -> None:
    pools = pool_map()
    rows  = []

    for in_path in sorted(LBL_DIR.glob("*_input.json")):
        jd_id    = in_path.stem[:-len("_input")]
        out_path = LBL_DIR / f"{jd_id}_output.json"
        if not out_path.exists():
            continue

        inp  = json.loads(in_path.read_text())
        outp = json.loads(out_path.read_text())
        labels = outp.get("labels", [])

        n_in  = len(inp)
        n_out = len(labels)
        delta = n_out - n_in

        lengths = [len(r["text"]) for r in inp]
        mean_len   = sum(lengths) / len(lengths) if lengths else 0
        median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0
        max_len    = max(lengths) if lengths else 0

        tag_counts = {t: 0 for t in TAGS}
        bad_tag = 0
        for r in labels:
            t = r.get("tag")
            if t in tag_counts:
                tag_counts[t] += 1
            else:
                bad_tag += 1

        # Auto-filled lines are detected from the joined CSV if present.
        csv_path = LBL_DIR / f"{jd_id}.csv"
        n_auto = 0
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path, keep_default_na=False)
            if "auto_filled" in df_csv.columns:
                n_auto = (df_csv["auto_filled"] == "Y").sum()

        row = {
            "jd_id":        jd_id,
            "pool":         pools.get(jd_id, "unknown"),
            "n_input":      n_in,
            "n_output":     n_out,
            "delta":        delta,
            "mean_len":     round(mean_len, 1),
            "median_len":   median_len,
            "max_len":      max_len,
            "n_auto":       int(n_auto),
            "bad_tag":      bad_tag,
        }
        row.update(tag_counts)
        # Percentages
        if n_out > 0:
            for t in TAGS:
                row[f"{t}_pct"] = round(100 * tag_counts[t] / n_out, 1)
        else:
            for t in TAGS:
                row[f"{t}_pct"] = 0.0
        rows.append(row)

    if not rows:
        raise SystemExit(f"No labelled JDs found in {LBL_DIR}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["pool", "jd_id"]).reset_index(drop=True)
    df.to_csv(OUT_PER_JD, index=False)
    print(f"✓ Per-JD table → {OUT_PER_JD.relative_to(ROOT)}  ({len(df)} JDs)\n")

    # ── Per-JD table (printed to stdout) ────────────────────────────────────
    show_cols = ["jd_id", "pool", "n_input", "n_output", "delta",
                 "mean_len", "n_auto"] + TAGS
    print("Per-JD descriptive table (counts + auto-fill + delta):")
    with pd.option_context("display.max_rows", None,
                           "display.max_colwidth", 50,
                           "display.width", 200):
        print(df[show_cols].to_string(index=False))
    print()

    # ── Cross-corpus summary ────────────────────────────────────────────────
    print("=" * 78)
    print("Cross-corpus descriptive summary")
    print("=" * 78)

    num_cols = ["n_input", "n_output", "mean_len", "median_len", "max_len",
                "n_auto"] + TAGS
    summary = df[num_cols].agg(["mean", "median", "std", "min", "max"]).round(1)
    print("\nPer-JD distributions:")
    print(summary.to_string())

    print(f"\nTotal JDs labelled: {len(df)}")
    print(f"JDs with delta != 0 (after autofill): {(df['delta'] != 0).sum()}  "
          "(should be 0)")
    print(f"JDs with bad_tag > 0: {(df['bad_tag'] > 0).sum()}  (should be 0)")
    print(f"JDs with at least one auto-filled line: {(df['n_auto'] > 0).sum()}")
    print(f"Total auto-filled lines across corpus: {df['n_auto'].sum()}")

    # By pool
    if "pool" in df.columns and df["pool"].nunique() > 1:
        print("\nBy JD pool:")
        by_pool = df.groupby("pool")[["n_input"] + TAGS].sum()
        by_pool["jds"] = df.groupby("pool").size()
        by_pool = by_pool[["jds", "n_input"] + TAGS]
        # Add share rows
        share = by_pool[TAGS].div(by_pool["n_input"], axis=0).round(3) * 100
        share.columns = [f"{c}_%" for c in TAGS]
        by_pool = pd.concat([by_pool, share], axis=1)
        print(by_pool.to_string())

    # Global tag distribution
    total_tags = {t: int(df[t].sum()) for t in TAGS}
    total = sum(total_tags.values())
    print("\nGlobal tag distribution:")
    for t in TAGS:
        pct = 100 * total_tags[t] / total if total else 0
        print(f"  {t:11s}: {total_tags[t]:>5d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':11s}: {total:>5d}")

    summary.to_csv(OUT_SUM)
    print(f"\n✓ Cross-corpus summary → {OUT_SUM.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
