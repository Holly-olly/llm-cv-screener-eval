#!/usr/bin/env python3
"""
Prompt P0 — Minimal prompt without rubric (floor baseline)

Purpose: Measure how much the structured rubric in the app actually contributes.
If P0 ≈ V1 → the rubric adds nothing. If P0 < V1 → rubric helps.

V1 (labels.json baseline) also had no rubric — this confirms that reading.
P0 is a clean, minimal prompt with no scoring instructions beyond the 4-band scale.

Output: data/batch-results/prompt_p0.json
Fields: jd_id, human_relevance, v1_score, p0_score, delta_vs_v1, verdict, summary

Usage:
    cd evaluation/
    python3 prompt_p0_simple.py
    python3 prompt_p0_simple.py --icc   # also run 5-JD ICC test (~10 extra calls)
"""

import json
import os
import re
import sys
import time
from pathlib import Path
import google.generativeai as genai

GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'


PROMPT_P0 = """You are a hiring manager. Assess how well this candidate fits this job.

CANDIDATE CV:
{cv}

JOB DESCRIPTION:
{jd}

Score bands:
- 75-100: Strong fit — would shortlist for interview
- 50-74:  Partial fit — real overlap but notable gaps
- 25-49:  Weak fit — adjacent domain, significant misalignment
- 0-24:   Not relevant — different field or hard dealbreaker

Respond in exactly this format — no other text:
SCORE: [0-100]
VERDICT: [Apply|Consider|Skip]
SUMMARY: [2-3 sentences on the match]

Rules:
- VERDICT Apply = score >= 70
- VERDICT Consider = score 45-69
- VERDICT Skip = score < 45"""


def load_api_key():
    env_path = Path(__file__).parent.parent / '.env.local'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY=') or line.startswith('VITE_GEMINI_API_KEY='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('GEMINI_API_KEY') or os.environ.get('VITE_GEMINI_API_KEY')


def parse_response(text):
    score, verdict, summary = None, '', ''
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('SCORE:'):
            m = re.search(r'\d+', line.split(':', 1)[1])
            if m: score = int(m.group())
        elif line.startswith('VERDICT:'):
            verdict = line.split(':', 1)[1].strip().split()[0]
        elif line.startswith('SUMMARY:'):
            summary = line.split(':', 1)[1].strip()
    return score, verdict, summary


def metrics(scores, human_labels, threshold=50):
    tp = tn = fp = fn = 0
    for s, h in zip(scores, human_labels):
        ai_bin = 1 if s >= threshold else 0
        h_bin  = 1 if h >= 2 else 0
        if ai_bin == 1 and h_bin == 1: tp += 1
        elif ai_bin == 0 and h_bin == 0: tn += 1
        elif ai_bin == 1 and h_bin == 0: fp += 1
        else: fn += 1
    neg = tp + tn + fp + fn
    return dict(acc=(tp+tn)/neg, fpr=fp/(fp+tn) if (fp+tn) else 0,
                fnr=fn/(fn+tp) if (fn+tp) else 0, fp=fp, fn=fn)


def pearson_r(xs, ys):
    n = len(xs)
    xm, ym = sum(xs)/n, sum(ys)/n
    num = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
    den = (sum((x-xm)**2 for x in xs) * sum((y-ym)**2 for y in ys)) ** 0.5
    return num/den if den else 0


def main():
    run_icc = '--icc' in sys.argv

    api_key = load_api_key()
    if not api_key:
        print('ERROR: No API key found. Set GEMINI_API_KEY or add to .env.local')
        return

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    cv_text = Path('evaluation/cv/cv_primary.txt').read_text()

    with open('evaluation/labels.json') as f:
        raw = json.load(f)
    labeled = [l for l in raw['labels'] if l.get('human_relevance') is not None and l.get('ai_score') is not None]

    print(f'Running P0 (minimal prompt) on {len(labeled)} JDs...')
    print(f'{"JD":<45} {"H":>3} {"V1":>5} {"P0":>5} {"Δ":>5}  {"Verdict"}')
    print('-' * 75)

    results = []
    for entry in labeled:
        jd_id   = entry['jd_id']
        jd_file = Path('evaluation') / entry['jd_file']
        if not jd_file.exists():
            print(f'  SKIP (missing): {jd_id}')
            continue

        jd_text = jd_file.read_text()
        human   = entry['human_relevance']
        v1      = entry['ai_score']

        prompt = PROMPT_P0.format(cv=cv_text, jd=jd_text)

        try:
            resp   = model.generate_content(prompt)
            score, verdict, summary = parse_response(resp.text)
            time.sleep(3)
        except Exception as e:
            err = str(e)
            print(f'{jd_id[:44]:<45} ERROR: {err[:60]}')
            if 'quota' in err.lower() or '429' in err:
                print('API limit — stopping.')
                break
            continue

        delta = (score - v1) if score is not None else None
        results.append({
            'jd_id':           jd_id,
            'human_relevance': human,
            'v1_score':        v1,
            'p0_score':        score,
            'delta_vs_v1':     delta,
            'verdict':         verdict,
            'summary':         summary,
        })
        delta_str = f'{delta:+d}' if delta is not None else '?'
        print(f'{jd_id[:44]:<45} {human:>3} {v1:>5.0f} {str(score or "?"):>5} {delta_str:>5}  {verdict}')

    # ── ICC re-test on 5 JDs ─────────────────────────────────────────────────
    icc_results = []
    if run_icc and results:
        icc_jds = [r for r in results[:5]]  # first 5 available
        print(f'\n--- ICC re-run on {len(icc_jds)} JDs ---')
        for r in icc_jds:
            entry = next((l for l in labeled if l['jd_id'] == r['jd_id']), None)
            if not entry: continue
            jd_text = (Path('evaluation') / entry['jd_file']).read_text()
            prompt  = PROMPT_P0.format(cv=cv_text, jd=jd_text)
            try:
                resp2 = model.generate_content(prompt)
                s2, _, _ = parse_response(resp2.text)
                time.sleep(3)
                diff = abs(r['p0_score'] - s2) if (r['p0_score'] and s2) else None
                print(f"  {r['jd_id'][:44]:<45} run1={r['p0_score']}  run2={s2}  diff={diff}")
                icc_results.append({'jd_id': r['jd_id'], 'run1': r['p0_score'], 'run2': s2, 'diff': diff})
            except Exception as e:
                print(f'  ICC ERROR: {e}')

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path('evaluation/data/batch-results/prompt_p0.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in results if r['p0_score'] is not None]
    if valid:
        p0_scores  = [r['p0_score'] for r in valid]
        v1_scores  = [r['v1_score'] for r in valid]
        humans     = [r['human_relevance'] for r in valid]

        m_p0 = metrics(p0_scores, humans)
        m_v1 = metrics(v1_scores, humans)
        r_p0 = pearson_r(p0_scores, humans)
        r_v1 = pearson_r(v1_scores, humans)

        print(f'\n{"":22} {"V1 baseline":>12} {"P0 simple":>12}')
        print('-' * 50)
        for key, label in [('acc','Accuracy'),('fpr','FP Rate'),('fnr','FN Rate'),('fp','FP count'),('fn','FN count')]:
            fmt = lambda v: f'{v:.0%}' if isinstance(v, float) else str(v)
            print(f'{label:<22} {fmt(m_v1[key]):>12} {fmt(m_p0[key]):>12}')
        print(f'{"r vs human":<22} {r_v1:>12.3f} {r_p0:>12.3f}')

        print('\nMean score by human label:')
        for h in [0,1,2,3]:
            sub_p0 = [r['p0_score'] for r in valid if r['human_relevance']==h]
            sub_v1 = [r['v1_score'] for r in valid if r['human_relevance']==h]
            if sub_p0:
                print(f'  H={h}: V1={sum(sub_v1)/len(sub_v1):.0f}  P0={sum(sub_p0)/len(sub_p0):.0f}  '
                      f'shift={sum(sub_p0)/len(sub_p0)-sum(sub_v1)/len(sub_v1):+.1f}')

        print(f'\nSaved to: {out_path}')

    if icc_results:
        icc_path = Path('data/batch-results/prompt_p0_icc.json')
        with open(icc_path, 'w') as f:
            json.dump(icc_results, f, indent=2)
        diffs = [r['diff'] for r in icc_results if r['diff'] is not None]
        if diffs:
            print(f'ICC mean diff: {sum(diffs)/len(diffs):.1f} pts  max: {max(diffs)} pts')
            print(f'Saved ICC to: {icc_path}')


if __name__ == '__main__':
    main()
