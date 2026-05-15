#!/usr/bin/env python3
"""
Build master dataset for Level 1: merge all per-prompt × per-CV JSON files
into a single tidy CSV. One row = one LLM call.

Inputs (any subset OK — missing files are skipped with a warning):
    results/level1_{p0,p1,p2}_{cv_primary,cv_hr,cv_engineer}.json

Output:
    results/level1_master.csv

This script is pure ETL — it does no analysis. All downstream analysis
(ICC, validity, cost) reads from level1_master.csv.
"""

import json
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / 'results'

PROMPTS = ['p0', 'p1', 'p2']
CVS     = ['cv_primary', 'cv_hr', 'cv_engineer']

COLUMNS = [
    'jd_id', 'cv', 'prompt', 'run_id', 'source',
    'model', 'temperature',
    'score', 'score_100', 'verdict',
    'human_holistic_label',
    'latency_s', 'prompt_tokens', 'output_tokens', 'total_tokens',
    'error',
]


def main():
    all_rows = []
    loaded, missing = 0, 0

    for prompt in PROMPTS:
        for cv in CVS:
            for suffix in ('', '_extra'):
                path = RESULTS / f'level1_{prompt}_{cv}{suffix}.json'
                if not path.exists():
                    if suffix == '':
                        print(f'  missing: {path.name}')
                        missing += 1
                    continue
                with open(path) as f:
                    rows = json.load(f)
                default_source = 'main' if suffix == '' else 'extra'
                for r in rows:
                    r.setdefault('prompt', prompt.upper())
                    r.setdefault('score_100', None)
                    r.setdefault('source', default_source)
                    all_rows.append(r)
                print(f'  loaded {len(rows):>4}  {path.name}')
                loaded += 1

    if not all_rows:
        print('\nNo data loaded.')
        return

    df = pd.DataFrame(all_rows)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[COLUMNS]

    out_path = RESULTS / 'level1_master.csv'
    df.to_csv(out_path, index=False)

    print(f'\nFiles loaded:  {loaded}')
    print(f'Files missing: {missing}')
    print(f'Total rows:    {len(df)}')
    print(f'\nBreakdown:')
    print(df.groupby(['prompt', 'cv']).size().unstack(fill_value=0).to_string())
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
