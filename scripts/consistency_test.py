#!/usr/bin/env python3
"""
Consistency Test (Test-Retest Reliability)
Runs 5 JDs through AI twice (same prompt, same CV) and measures ICC.

5 JDs chosen to span the score range:
  1. codility_assessment_scientist     — strong fit (AI~92)
  2. Johnson_Controls_People Insights  — partial fit (AI~78)
  3. freeplay_data_scientist           — borderline (AI~55)
  4. Arrive_People_Analytics_Lead      — adjacent/FP zone (AI~65)
  5. sorare-marketing-graphic-designer — out-of-domain (AI~0)

ICC(2,1) — two-way random, absolute agreement, single rater.
Target: ICC > 0.80

Usage:
    cd evaluation/
    python3 consistency_test.py

Results saved to: data/batch-results/consistency_icc.json
"""

import json
import os
import time
import numpy as np
from pathlib import Path
from google import genai

GEMINI_MODEL = 'gemini-3.1-flash-lite-preview'

# Same prompt as batch_analyze.py (V1)
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

TEST_JDS = [
    ('codility_assessment_scientist',          'data/batch-jobs/codility_assessment_scientist.txt',        3),
    ('Johnson_Controls_People Insights_Partner','data/batch-jobs/Johnson_Controls_People Insights_Partner .txt', 2),
    ('freeplay_data_scientist',                 'data/batch-jobs/freeplay_data_scientist.txt',              2),
    ('Arrive_People_Analytics_Lead',            'data/batch-jobs/Arrive_People_Analytics_Lead.txt',         1),
    ('sorare-marketing-graphic-designer',       'data/batch-jobs/sorare-marketing-graphic-designer.txt',    0),
]

N_REPS = 2  # Run each JD this many times


def load_api_key():
    env_path = Path(__file__).parent.parent / '.env.local'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY='):
                return line.split('=', 1)[1].strip()
    return os.environ.get('GEMINI_API_KEY')


def parse_score(text):
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith('SCORE:'):
            try:
                return int(''.join(c for c in line.split(':', 1)[1] if c.isdigit()))
            except:
                pass
    return None


def icc_2way_absolute(scores_matrix):
    """
    ICC(2,1) — two-way random effects, absolute agreement, single measures.
    scores_matrix: shape (n_subjects, n_raters/reps)
    Returns: icc, F, df1, df2, p
    """
    from scipy import stats
    k = scores_matrix.shape[1]  # number of reps
    n = scores_matrix.shape[0]  # number of subjects

    grand_mean = scores_matrix.mean()
    row_means  = scores_matrix.mean(axis=1)
    col_means  = scores_matrix.mean(axis=0)

    SSr = k * np.sum((row_means - grand_mean) ** 2)
    SSc = n * np.sum((col_means - grand_mean) ** 2)
    SSe = np.sum((scores_matrix - row_means[:, None] - col_means[None, :] + grand_mean) ** 2)
    SSt = np.sum((scores_matrix - grand_mean) ** 2)

    dfr = n - 1
    dfc = k - 1
    dfe = dfr * dfc

    MSr = SSr / dfr
    MSc = SSc / dfc
    MSe = SSe / dfe if dfe > 0 else 0

    # ICC(2,1) absolute agreement
    icc = (MSr - MSe) / (MSr + (k - 1) * MSe + k * (MSc - MSe) / n)
    icc = max(0.0, icc)  # floor at 0

    F = MSr / MSe if MSe > 0 else float('inf')
    p = 1 - stats.f.cdf(F, dfr, dfe) if MSe > 0 else 0.0

    return icc, F, dfr, dfe, p


def main():
    api_key = load_api_key()
    client = genai.Client(api_key=api_key)
    cv_text = Path('cv/cv_primary.txt').read_text()

    results = []
    print(f'Consistency Test — {len(TEST_JDS)} JDs × {N_REPS} runs each\n')
    print(f'{"JD":<45} {"H":>3}  ' + '  '.join([f'Run{i+1}' for i in range(N_REPS)]) + '  SD')
    print('-' * 70)

    for jd_id, jd_path, human in TEST_JDS:
        jd_file = Path(jd_path)
        if not jd_file.exists():
            print(f'  SKIP (missing): {jd_id}')
            continue

        jd_text = jd_file.read_text()
        scores = []

        for rep in range(N_REPS):
            try:
                prompt = PROMPT_TEMPLATE.format(cv=cv_text, jd=jd_text)
                response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
                score = parse_score(response.text)
                scores.append(score or 0)
                time.sleep(3)  # slightly longer delay between calls
            except Exception as e:
                err = str(e)
                print(f'  ERROR on {jd_id} run {rep+1}: {err[:60]}')
                if 'quota' in err.lower() or '429' in err:
                    print('API limit reached.')
                    break
                scores.append(None)

        valid = [s for s in scores if s is not None]
        sd = np.std(valid) if len(valid) > 1 else float('nan')
        scores_str = '  '.join([f'{s:5}' if s is not None else ' ERR' for s in scores])

        results.append({
            'jd_id': jd_id,
            'human_relevance': human,
            'scores': scores,
            'mean': float(np.mean(valid)) if valid else None,
            'sd': float(sd) if not np.isnan(sd) else None,
            'max_diff': int(max(valid) - min(valid)) if len(valid) > 1 else None,
        })

        print(f'{jd_id[:44]:<45} {human:>3}  {scores_str}  SD={sd:.1f}')

    # Save
    out_path = 'data/batch-results/consistency_icc.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Calculate ICC
    complete = [r for r in results if len([s for s in r['scores'] if s is not None]) == N_REPS]

    if len(complete) >= 2:
        matrix = np.array([[s for s in r['scores'] if s is not None] for r in complete])

        icc, F, df1, df2, p = icc_2way_absolute(matrix)

        print(f'\n=== CONSISTENCY SUMMARY ===')
        print(f'JDs tested:     {len(complete)}')
        print(f'Reps per JD:    {N_REPS}')
        print()
        print(f'{"JD":<45} {"Mean":>6} {"SD":>6} {"MaxDiff":>8}')
        print('-' * 70)
        for r in complete:
            print(f'{r["jd_id"][:44]:<45} {r["mean"]:>6.1f} {r["sd"]:>6.1f} {r["max_diff"]:>8}')

        print()
        print(f'ICC(2,1):       {icc:.3f}  (target > 0.80)  {"✓" if icc > 0.80 else "✗"}')
        print(f'F({df1},{df2}):         {F:.2f}')
        print(f'p-value:        {p:.4f}')

        all_sds = [r['sd'] for r in complete if r['sd'] is not None]
        all_diffs = [r['max_diff'] for r in complete if r['max_diff'] is not None]
        print(f'Mean SD:        {np.mean(all_sds):.1f} pts')
        print(f'Max diff:       {max(all_diffs)} pts')
        print(f'\nSaved to: {out_path}')
    else:
        print('\nNot enough complete data to calculate ICC.')


if __name__ == '__main__':
    main()
