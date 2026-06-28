#!/usr/bin/env python3
"""Sandbox g21 — Level-1 holistic scoring on OpenAI (cross-provider replication).

Same L1-P2 prompt as the Gemini black-box runs (recruiter persona + rubric + 0–100
scale), ONLY the model changes: gpt-4o-mini instead of gemini-3.1-flash-lite. This
isolates the provider/model variable for the injection effect.

Conditions: clean + A/B/C/D (payloads identical to g16/g17), injection appended at CV
bottom, recruiter persona, temp=1.0, 1 run per JD, same 50 baseline JDs. No noise control.

Deltas are vs the OpenAI clean run (each provider gets its own baseline).
Writes: results/openai_l1_cv_primary.csv
"""
from __future__ import annotations
import importlib.util, csv, time
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]   # .../llm_evaluation
def _load(name, p):
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
g16 = _load("g16", HERE / "g16_injection_A.py")
g17 = _load("g17", HERE / "g17_injections_BCD.py")
l03 = _load("l03", ROOT / "scripts" / "level3" / "03_llm_label_json.py")

DATA, BASE, CV = g16.DATA, g16.BASE, g16.CV
PERSONA, BODY, parse, find_jd = g16.PERSONA, g16.BODY, g16.parse, g16.find_jd
OUT = BASE.parent / "openai_l1_cv_primary.csv"
MODEL, TEMP = "gpt-4o-mini", 1.0
CONDS = ["A", "B", "C", "D"]
PAYLOADS = {"A": g16.INJECTION_A, "B": g17.CONDITIONS["B"],
            "C": g17.CONDITIONS["C"], "D": g17.CONDITIONS["D"]}


def call(client, prompt):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=TEMP,
                messages=[{"role": "user", "content": prompt}])
            return parse(r.choices[0].message.content)
        except Exception as ex:
            if attempt < 3:
                time.sleep(8); continue
            return None, f"ERR:{str(ex)[:20]}"
    return None, "ERR"


def main():
    client = OpenAI(api_key=l03.load_openai_key())
    base = list(csv.DictReader(open(BASE, encoding="utf-8")))
    cv_clean = (DATA / "cv" / f"{CV}.txt").read_text().rstrip()
    cv_variants = {"clean": cv_clean}
    for c in CONDS:
        cv_variants[c] = cv_clean + "\n\n" + PAYLOADS[c] + "\n"

    print(f"model={MODEL} temp={TEMP}  JDs={len(base)}  conds=clean+{CONDS}\n")
    rows = []
    for i, r in enumerate(base, 1):
        jd_id = r["jd_id"].strip()
        jp = find_jd(jd_id)
        if not jp:
            print(f"  [{i}/{len(base)}] {jd_id[:36]} MISSING txt"); continue
        jd_text = jp.read_text()
        rec = {"jd_id": jd_id, "pool": r.get("pool", ""),
               "gemini_clean_100": r.get("recruiter_100", ""), "glass_100": r.get("glass_100", "")}
        scores = {}
        for cond, cv in cv_variants.items():
            s, v = call(client, PERSONA + BODY.format(cv=cv, jd=jd_text))
            scores[cond] = s
            rec[f"oai_{cond}_100"] = s
            rec[f"oai_{cond}_verdict"] = v
            time.sleep(0.4)
        cl = scores.get("clean")
        for c in CONDS:
            rec[f"oai_{c}_delta"] = (scores[c] - cl) if (scores[c] is not None and cl is not None) else None
        rows.append(rec)
        ds = "  ".join(f"{c}:{rec[f'oai_{c}_delta']:+d}" if rec[f'oai_{c}_delta'] is not None else f"{c}:na"
                       for c in CONDS)
        print(f"  [{i}/{len(base)}] {jd_id[:30]:<30} clean={str(cl):>4}  {ds}")

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # summary
    def col(c):
        return [r[f"oai_{c}_delta"] for r in rows if r.get(f"oai_{c}_delta") is not None]
    print("\n" + "=" * 56)
    print(f"OpenAI L1 ({MODEL}) injection deltas vs OpenAI clean, n JDs:")
    print(f"{'cond':<6}{'mean Δ':>9}{'median':>8}{'max':>6}{'infl>0':>9}")
    for c in CONDS:
        d = col(c)
        if not d:
            continue
        sd = sorted(d)
        med = sd[len(sd)//2]
        print(f"{c:<6}{sum(d)/len(d):>+9.1f}{med:>+8d}{max(d):>+6d}{sum(1 for x in d if x>0):>6}/{len(d)}")
    print("=" * 56)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
