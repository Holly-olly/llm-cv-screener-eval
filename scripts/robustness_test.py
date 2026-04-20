#!/usr/bin/env python3
"""
CV Robustness Test
Runs same 24 JDs through AI with primary CV vs paraphrased CV.
Target: score shift < 5 pts for 80%+ of roles.

Usage:
    python3 robustness_test.py
"""

import json
import os
import time
from pathlib import Path
from google import genai

PROMPT_TEMPLATE = """You are a senior talent acquisition expert evaluating candidate fit.

=== CANDIDATE CV ===
{cv}

=== JOB DESCRIPTION ===
{jd}

=== INSTRUCTIONS ===
Respond in exactly this format — no other text:

SCORE: [number 0-100]
VERDICT: [Apply | Consider | Skip]
SUMMARY: [2-3 sentences explaining the score]

Rules:
- SCORE 75-100 = strong fit
- SCORE 50-74 = partial fit
- SCORE 25-49 = weak fit
- SCORE 0-24 = not relevant
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
    result = {'score': None, 'verdict': ''}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('SCORE:'):
            try: result['score'] = int(''.join(c for c in line.split(':',1)[1] if c.isdigit()))
            except: pass
        elif line.startswith('VERDICT:'):
            result['verdict'] = line.split(':',1)[1].strip()
    return result

def call_gemini(client, prompt):
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',
        contents=prompt
    )
    return response.text

def main():
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    with open('labels.json') as f:
        labels_raw = json.load(f)
    labeled = [l for l in labels_raw['labels'] if l.get('ai_score') is not None]

    cv_primary = Path('cv/cv_primary.txt').read_text()
    cv_paraphrased = Path('cv/cv_paraphrased.txt').read_text()

    results = []
    print(f'CV Robustness Test — {len(labeled)} JDs\n')
    print(f'{"JD":<45} {"Prim":>5} {"Para":>5} {"Diff":>5} {"Stable"}')
    print('-' * 70)

    for i, entry in enumerate(labeled, 1):
        jd_file = Path(entry['jd_file'])
        if not jd_file.exists():
            print(f'  SKIP (file missing): {entry["jd_id"]}')
            continue

        jd_text = jd_file.read_text()
        jd_id = entry['jd_id'][:44]

        try:
            r1 = parse_response(call_gemini(client, PROMPT_TEMPLATE.format(cv=cv_primary, jd=jd_text)))
            time.sleep(2)
            r2 = parse_response(call_gemini(client, PROMPT_TEMPLATE.format(cv=cv_paraphrased, jd=jd_text)))
            time.sleep(2)

            s1 = r1['score'] or 0
            s2 = r2['score'] or 0
            diff = abs(s1 - s2)
            stable = '✓' if diff <= 5 else ('△' if diff <= 10 else '✗')

            results.append({
                'jd_id': entry['jd_id'],
                'human_relevance': entry.get('human_relevance'),
                'original_ai_score': entry.get('ai_score'),
                'primary_score': s1,
                'paraphrased_score': s2,
                'diff': diff,
                'stable': diff <= 5
            })
            print(f'{jd_id:<45} {s1:>5} {s2:>5} {diff:>+5} {stable}')

        except Exception as e:
            err = str(e)
            print(f'{jd_id:<45} ERROR: {err[:50]}')
            if 'quota' in err.lower() or '429' in err or 'limit' in err.lower():
                print(f'\n⛔ API limit reached after {i-1} files. Resume tomorrow.')
                break
            continue

    # Save
    out_path = 'data/batch-results/robustness_cv_paraphrase.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    if results:
        diffs = [r['diff'] for r in results]
        stable_n = sum(1 for r in results if r['stable'])
        pct = stable_n / len(results)
        print(f'\n=== ROBUSTNESS SUMMARY ===')
        print(f'Tested:          {len(results)}')
        print(f'Stable (≤5 pts): {stable_n}/{len(results)} = {pct:.0%}  (target > 80%)  {"✓" if pct > 0.80 else "✗"}')
        print(f'Mean diff:       {sum(diffs)/len(diffs):.1f} pts')
        print(f'Max diff:        {max(diffs)} pts')
        unstable = [r for r in results if not r['stable']]
        if unstable:
            print(f'\nUnstable JDs:')
            for r in unstable:
                print(f'  {r["jd_id"]}: {r["primary_score"]} → {r["paraphrased_score"]} (diff={r["diff"]})')
        print(f'\nSaved to: {out_path}')

if __name__ == '__main__':
    main()
