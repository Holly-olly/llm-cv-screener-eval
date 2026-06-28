#!/usr/bin/env python3
"""Sandbox g22 — assemble all injection scores into one master file, keyed by JD.

Merges, per JD (50 baseline JDs, cv_primary):
  glass_100                  — Level-3 glass-box (constant; injection-immune)
  gem_clean, gem_{A..D}      — Level-1 holistic, Gemini  (recruiter persona)
  oai_clean, oai_{A..D}      — Level-1 holistic, OpenAI gpt-4o-mini (same prompt)
  + per-condition deltas vs each provider's own clean.

Reads:  injection_all_cv_primary.csv (Gemini), openai_l1_cv_primary.csv (OpenAI)
Writes: results/master_injection_scores_cv_primary.csv
"""
from __future__ import annotations
import csv
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
GEM = R / "injection_all_cv_primary.csv"
OAI = R / "openai_l1_cv_primary.csv"
OUT = R / "master_injection_scores_cv_primary.csv"
CONDS = ["A", "B", "C", "D"]


def num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def main():
    gem = {r["jd_id"].strip(): r for r in csv.DictReader(open(GEM, encoding="utf-8"))}
    oai = {r["jd_id"].strip(): r for r in csv.DictReader(open(OAI, encoding="utf-8"))}

    rows = []
    for jid, g in gem.items():
        o = oai.get(jid, {})
        rec = {"jd_id": jid, "pool": g.get("pool", ""), "glass_100": num(g.get("glass_100"))}
        # Gemini L1
        rec["gem_clean"] = num(g.get("recruiter_100"))
        for c in CONDS:
            rec[f"gem_{c}"] = num(g.get(f"inj{c}_100"))
            rec[f"gem_d{c}"] = num(g.get(f"inj{c}_delta"))
        # OpenAI L1
        rec["oai_clean"] = num(o.get("oai_clean_100"))
        for c in CONDS:
            rec[f"oai_{c}"] = num(o.get(f"oai_{c}_100"))
            rec[f"oai_d{c}"] = num(o.get(f"oai_{c}_delta"))
        rows.append(rec)

    cols = (["jd_id", "pool", "glass_100", "gem_clean"]
            + [f"gem_{c}" for c in CONDS] + [f"gem_d{c}" for c in CONDS]
            + ["oai_clean"] + [f"oai_{c}" for c in CONDS] + [f"oai_d{c}" for c in CONDS])
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    print(f"wrote {OUT}  ({len(rows)} JDs, {len(cols)} cols)")
    # quick integrity: per-provider mean delta
    def md(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return sum(v) / len(v) if v else float("nan")
    print("\nmean Δ recap (sanity):")
    print(f"{'cond':<6}{'Gemini':>9}{'OpenAI':>9}")
    for c in CONDS:
        print(f"{c:<6}{md(f'gem_d{c}'):>+9.1f}{md(f'oai_d{c}'):>+9.1f}")


if __name__ == "__main__":
    main()
