#!/usr/bin/env python3
"""
Synthetic Control CV Test — Discrimination / Specificity Check

Runs the synthetic UX researcher CV against all 24 labelled JDs.
Expected: all scores should be low (< 40) especially for psychometrics roles.
High score on a psychometrics JD would mean the AI doesn't discriminate properly.

Results saved to: data/batch-results/synthetic_cv_scores.json

Usage:
    cd evaluation/
    python3 synthetic_cv_test.py
"""

import json
import os
import time
from pathlib import Path
from google import genai

GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'

PROMPT_TEMPLATE = """You are a senior talent acquisition expert evaluating candidate fit.

=== CANDIDATE CV ===
{cv}

=== JOB DESCRIPTION ===
{jd}

=== INSTRUCTIONS ===
Evaluate how well this candidate fits this job description.

Respond in exactly this format — no other text:

SCORE: [number 0-100]
VERDICT: [Apply | Consider | Skip]
SUMMARY: [2-3 sentences explaining the score. Be specific about matches and gaps.]

Rules:
- SCORE 75-100 = strong fit, candidate has core required skills
- SCORE 50-74 = partial fit, some relevant skills but notable gaps
- SCORE 25-49 = weak fit, adjacent but significant misalignment
- SCORE 0-24 = not relevant, different domain or dealbreaker present
- VERDICT Apply = score >= 70
- VERDICT Consider = score 45-69
- VERDICT Skip = score < 45
"""


def load_api_key():
    env_path = Path(__file__).parent.parent / '.env.local'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('GEMINI_API_KEY')


def parse_response(text):
    result = {'score': None, 'verdict': '', 'summary': ''}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('SCORE:'):
            try:
                result['score'] = int(''.join(c for c in line.split(':', 1)[1] if c.isdigit()))
            except:
                pass
        elif line.startswith('VERDICT:'):
            result['verdict'] = line.split(':', 1)[1].strip()
        elif line.startswith('SUMMARY:'):
            result['summary'] = line.split(':', 1)[1].strip()
    return result


def main():
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)
    cv_synth = Path('cv/cv_synthetic_control.txt').read_text()

    with open('labels.json') as f:
        raw = json.load(f)
    labeled = [l for l in raw['labels'] if l.get('human_relevance') is not None]

    results = []
    print('Synthetic CV Test — UX Researcher vs 24 JDs\n')
    print(f'{"JD":<45} {"H":>3} {"Primary":>8} {"Synth":>6}  Flag')
    print('-' * 75)

    for i, entry in enumerate(labeled, 1):
        jd_file = Path(entry['jd_file'])
        if not jd_file.exists():
            continue

        jd_text = jd_file.read_text()
        jd_id = entry['jd_id'][:44]
        human = entry.get('human_relevance', '?')
        primary_score = entry.get('ai_score', '?')

        try:
            r = parse_response(
                client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=PROMPT_TEMPLATE.format(cv=cv_synth, jd=jd_text)
                ).text
            )
            time.sleep(2)

            s = r['score'] or 0

            # Flag if synthetic scores suspiciously high on a psychometrics role
            flag = ''
            if human >= 2 and s >= 40:
                flag = ' ← HIGH (expected < 40)'
            if human >= 3 and s >= 50:
                flag = ' ← VERY HIGH (AI not discriminating!)'

            results.append({
                'jd_id': entry['jd_id'],
                'human_relevance': human,
                'primary_score': primary_score,
                'synthetic_score': s,
                'synthetic_verdict': r['verdict'],
                'synthetic_summary': r['summary'],
            })
            print(f'{jd_id:<45} {human:>3} {str(primary_score):>8} {s:>6}  {r["verdict"]}{flag}')

        except Exception as e:
            err = str(e)
            print(f'{jd_id:<45} ERROR: {err[:50]}')
            if 'quota' in err.lower() or '429' in err:
                print(f'\nAPI limit reached after {i-1} files.')
                break
            continue

    out_path = 'data/batch-results/synthetic_cv_scores.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        import numpy as np
        synth_scores = [r['synthetic_score'] for r in results]
        primary_scores = [r['primary_score'] for r in results if isinstance(r['primary_score'], (int, float))]

        print(f'\n=== DISCRIMINATION SUMMARY ===')
        print(f'Synthetic CV mean score:  {np.mean(synth_scores):.1f}  (expected < 25)')
        print(f'Primary CV mean score:    {np.mean(primary_scores):.1f}')
        print(f'Synthetic max score:      {max(synth_scores)}')
        print(f'Synthetic scores >= 40:   {sum(1 for s in synth_scores if s >= 40)}  (expected: 0)')
        print(f'Synthetic scores >= 50:   {sum(1 for s in synth_scores if s >= 50)}  (expected: 0)')

        # By human label
        print()
        for h in [3, 2, 1, 0]:
            sub = [r for r in results if r['human_relevance'] == h]
            if sub:
                mean_s = np.mean([r['synthetic_score'] for r in sub])
                mean_p = np.mean([r['primary_score'] for r in sub if isinstance(r['primary_score'], (int, float))])
                print(f'Human={h}:  Primary avg={mean_p:.0f}  Synthetic avg={mean_s:.0f}  Gap={mean_p-mean_s:.0f} pts')

        print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()
