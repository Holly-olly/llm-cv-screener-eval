#!/usr/bin/env python3
"""Sandbox g20 — noise control: separate the injection effect from run-to-run jitter.

Black-box recruiter persona is sampled at temperature=1.0, so the SAME input gives a
slightly different score each call. With 1 run per JD (g16/g17) part of each Δ is just
that jitter. Here we repeat each condition N times on a JD subset, so we can compare:
  within-condition SD  (jitter on identical input)   vs   between-condition shift (effect).

Subset: REPEATS runs of {clean, A, B, C, D} on N_JD JDs spread across the baseline range.
Conditions reuse the exact payloads from g16 (A) and g17 (B/C/D); `clean` = no injection.

Writes: results/noise_runs_cv_primary.csv  (long: jd_id, cond, run, score_100, verdict)
"""
from __future__ import annotations
import importlib.util, csv, time
from pathlib import Path
import statistics as st
import google.generativeai as genai

HERE = Path(__file__).resolve().parent
def _load(name, p):
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
g16 = _load("g16", HERE / "g16_injection_A.py")
g17 = _load("g17", HERE / "g17_injections_BCD.py")

DATA, BASE, CV = g16.DATA, g16.BASE, g16.CV
PERSONA, BODY = g16.PERSONA, g16.BODY
parse, find_jd, load_key = g16.parse, g16.find_jd, g16.load_key
OUT = BASE.parent / "noise_runs_cv_primary.csv"

PAYLOADS = {"clean": None, "A": g16.INJECTION_A,
            "B": g17.CONDITIONS["B"], "C": g17.CONDITIONS["C"], "D": g17.CONDITIONS["D"]}
N_JD, REPEATS, BASE_SLEEP = 8, 5, 4.0


def pick_subset(base_rows):
    """N_JD JDs spread evenly across the baseline recruiter_100 range."""
    valid = [r for r in base_rows if r.get("recruiter_100") not in (None, "", "None")
             and find_jd(r["jd_id"].strip())]
    valid.sort(key=lambda r: int(r["recruiter_100"]))
    idx = [round(i * (len(valid) - 1) / (N_JD - 1)) for i in range(N_JD)]
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i); out.append(valid[i])
    return out


def call(text):
    for attempt in range(4):
        try:
            cli = genai.GenerativeModel(g16.MODEL,
                                        generation_config=genai.GenerationConfig(temperature=g16.TEMP))
            return parse(cli.generate_content(text).text)
        except Exception as ex:
            if "429" in str(ex) and attempt < 3:
                time.sleep(20); continue
            return None, f"ERR:{str(ex)[:20]}"
    return None, "ERR"


def main():
    genai.configure(api_key=load_key())
    base = list(csv.DictReader(open(BASE, encoding="utf-8")))
    subset = pick_subset(base)
    cv_clean = (DATA / "cv" / f"{CV}.txt").read_text().rstrip()
    print(f"subset ({len(subset)} JDs by baseline spread): "
          f"{[ (r['jd_id'].strip()[:18], r['recruiter_100']) for r in subset ]}\n")

    rows = []
    for r in subset:
        jd_id = r["jd_id"].strip()
        jd_text = find_jd(jd_id).read_text()
        for cond, payload in PAYLOADS.items():
            cv = cv_clean if payload is None else cv_clean + "\n\n" + payload + "\n"
            for run in range(1, REPEATS + 1):
                s, v = call(PERSONA + BODY.format(cv=cv, jd=jd_text))
                rows.append({"jd_id": jd_id, "cond": cond, "run": run, "score_100": s, "verdict": v})
                time.sleep(BASE_SLEEP)
            scores = [x["score_100"] for x in rows if x["jd_id"] == jd_id and x["cond"] == cond
                      and x["score_100"] is not None]
            m = st.mean(scores) if scores else float("nan")
            sd = st.pstdev(scores) if len(scores) > 1 else 0.0
            print(f"  {jd_id[:30]:<30} {cond:<6} mean={m:5.1f} sd={sd:4.1f}  runs={scores}")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- aggregate: within-condition SD vs between-condition shift ----
    def cell(jd, cond):
        return [x["score_100"] for x in rows if x["jd_id"] == jd and x["cond"] == cond
                and x["score_100"] is not None]
    jds = [r["jd_id"].strip() for r in subset]
    print("\n" + "=" * 64)
    print("Pooled within-condition SD (jitter on identical input):")
    for cond in PAYLOADS:
        sds = [st.pstdev(cell(j, cond)) for j in jds if len(cell(j, cond)) > 1]
        print(f"  {cond:<6} mean within-JD SD = {st.mean(sds):.2f}")
    print("\nMean effect vs clean (per-JD mean shift), pooled over subset:")
    base_means = {j: st.mean(cell(j, "clean")) for j in jds if cell(j, "clean")}
    for cond in ["A", "B", "C", "D"]:
        diffs = [st.mean(cell(j, cond)) - base_means[j] for j in jds
                 if cell(j, cond) and j in base_means]
        print(f"  {cond:<6} mean Δ = {st.mean(diffs):+.1f}   (per-JD Δ: "
              f"{[round(d) for d in diffs]})")
    print("=" * 64)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
