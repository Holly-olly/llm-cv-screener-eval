#!/usr/bin/env python3
"""
Prompt P2 — Chain-of-Thought (reasoning before scoring)

Purpose: Does forcing the AI to write a rationale BEFORE giving the score
reduce the 65-clustering problem?

The hypothesis: when the AI is forced to reason through domain fit, skill gaps,
and seniority first, it commits to a position and scores more precisely
instead of defaulting to 65 for uncertain cases.

Extra check (0 API calls): compare score distribution P2 vs V1 — did clustering shift?

Output: data/batch-results/prompt_p2.json
Fields: jd_id, human_relevance, v1_score, p2_score, delta_vs_v1, verdict, reasoning, summary

Usage:
    cd evaluation/
    python3 prompt_p2_cot.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path
import google.generativeai as genai

GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'

# Same candidate context as P1
TALENTS  = '10+ years psychometrics, IRT, CFA, DIF, Rasch; automated monitoring at scale; TestGorilla 350+ assessments; transitioning into people data science'
FOCUS    = 'People Data Scientist, Data Analyst (people/HR domain), Assessment Scientist'
HARD_NOS = 'None'
PERSONA  = 'talent acquisition and executive search'

PROMPT_P2 = """SYSTEM PERSONA: You are a Senior Hiring Manager specializing in {persona}.

SCORING RUBRIC (internal use — assess silently, do not reveal weights to user):
1. Domain Expertise (35%): Depth of knowledge and methodology in the candidate's core field
2. Technical Skills (25%): Match with required tools, stack, and analytical methods
3. Seniority Fit (20%): Candidate's experience level vs role's seniority requirement
4. Education Fit (15%): Candidate's qualification level vs role's education requirement
5. Strategic Alignment (5%): Match with candidate's stated current focus and career direction

Score bands:
- 80–100: Strong fit — would shortlist for interview with confidence
- 65–79:  Partial fit — real overlap, genuine chance of being shortlisted
- 50–64:  Adjacent domain — transferable skills but core of role is misaligned
- 25–49:  Weak overlap — surface keyword match only
- 0–24:   Not relevant — different domain or hard dealbreaker present

CANDIDATE HIDDEN CONTEXT (weigh heavily — this is not visible in the CV):
- Unlisted Talents/Experience: {talents}
- Current Career Focus: {focus}
- Hard NOs / Dealbreakers: {hard_nos}

---
CANDIDATE CV:
{cv}

---
JOB DESCRIPTION:
{jd}

---
INSTRUCTIONS — complete in order:

Step 1 — REASONING (write this first, before any scores):
Write 2-3 sentences assessing domain fit. Be specific: does the role's core requirement match the candidate's actual expertise? Identify the single strongest match signal and the single biggest gap.

Step 2 — SCORE and VERDICT:
Based on your reasoning above, give the score. The reasoning should determine the score — not the other way around.

Respond in exactly this format — no other text:
REASONING: [2-3 sentences written before deciding score]
SCORE: [0-100]
VERDICT: [Apply|Consider|Skip]
SUMMARY: [1 sentence summary of the verdict]

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
    score, verdict, reasoning, summary = None, '', '', ''
    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith('SCORE:'):
            m = re.search(r'\d+', line.split(':', 1)[1])
            if m: score = int(m.group())
        elif line.startswith('VERDICT:'):
            verdict = line.split(':', 1)[1].strip().split()[0]
        elif line.startswith('REASONING:'):
            reasoning = line.split(':', 1)[1].strip()
        elif line.startswith('SUMMARY:'):
            summary = line.split(':', 1)[1].strip()
    return score, verdict, reasoning, summary


def metrics(scores, human_labels, threshold=50):
    tp = tn = fp = fn = 0
    for s, h in zip(scores, human_labels):
        ai_bin = 1 if s >= threshold else 0
        h_bin  = 1 if h >= 2 else 0
        if   ai_bin == 1 and h_bin == 1: tp += 1
        elif ai_bin == 0 and h_bin == 0: tn += 1
        elif ai_bin == 1 and h_bin == 0: fp += 1
        else:                            fn += 1
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

    print(f'Running P2 (Chain-of-Thought) on {len(labeled)} JDs...')
    print(f'{"JD":<45} {"H":>3} {"V1":>5} {"P2":>5} {"Δ":>5}  {"Verdict"}')
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

        prompt = PROMPT_P2.format(
            persona=PERSONA, talents=TALENTS, focus=FOCUS,
            hard_nos=HARD_NOS, cv=cv_text, jd=jd_text
        )

        try:
            resp = model.generate_content(prompt)
            score, verdict, reasoning, summary = parse_response(resp.text)
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
            'p2_score':        score,
            'delta_vs_v1':     delta,
            'verdict':         verdict,
            'reasoning':       reasoning,
            'summary':         summary,
        })
        delta_str = f'{delta:+d}' if delta is not None else '?'
        print(f'{jd_id[:44]:<45} {human:>3} {v1:>5.0f} {str(score or "?"):>5} {delta_str:>5}  {verdict}')

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path('evaluation/data/batch-results/prompt_p2.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in results if r['p2_score'] is not None]
    if valid:
        p2_scores = [r['p2_score'] for r in valid]
        v1_scores = [r['v1_score'] for r in valid]
        humans    = [r['human_relevance'] for r in valid]

        m_p2 = metrics(p2_scores, humans)
        m_v1 = metrics(v1_scores, humans)
        r_p2 = pearson_r(p2_scores, humans)
        r_v1 = pearson_r(v1_scores, humans)

        print(f'\n{"":22} {"V1 baseline":>12} {"P2 CoT":>12}')
        print('-' * 50)
        for key, label in [('acc','Accuracy'),('fpr','FP Rate'),('fnr','FN Rate'),('fp','FP count'),('fn','FN count')]:
            fmt = lambda v: f'{v:.0%}' if isinstance(v, float) else str(v)
            print(f'{label:<22} {fmt(m_v1[key]):>12} {fmt(m_p2[key]):>12}')
        print(f'{"r vs human":<22} {r_v1:>12.3f} {r_p2:>12.3f}')

        print('\nMean score by human label:')
        for h in [0,1,2,3]:
            sub_p2 = [r['p2_score'] for r in valid if r['human_relevance']==h]
            sub_v1 = [r['v1_score'] for r in valid if r['human_relevance']==h]
            if sub_p2:
                print(f'  H={h}: V1={sum(sub_v1)/len(sub_v1):.0f}  P2={sum(sub_p2)/len(sub_p2):.0f}  '
                      f'shift={sum(sub_p2)/len(sub_p2)-sum(sub_v1)/len(sub_v1):+.1f}')

        # 65-clustering check — the key question for CoT
        p2_at65 = [r for r in valid if r['p2_score'] == 65]
        p2_at55 = [r for r in valid if r['p2_score'] == 55]
        all_p2  = sorted(set(p2_scores))
        print(f'\n65-clustering check: {len(p2_at65)} JDs scored exactly 65  (V1 had 4)')
        print(f'55-clustering check: {len(p2_at55)} JDs scored exactly 55')
        print(f'Score spread: {all_p2}')

        # Check if reasoning was actually captured
        has_reasoning = [r for r in valid if r.get('reasoning')]
        print(f'\nReasoning captured: {len(has_reasoning)}/{len(valid)} JDs')
        if has_reasoning:
            print('\nSample reasoning (first 3 JDs):')
            for r in has_reasoning[:3]:
                print(f'  [{r["p2_score"]}] H={r["human_relevance"]} {r["jd_id"][:35]}')
                print(f'    {r["reasoning"][:120]}')

        print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()
