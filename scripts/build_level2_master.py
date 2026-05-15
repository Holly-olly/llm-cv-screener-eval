#!/usr/bin/env python3
"""
Build level2_master.csv from all level2_p0_*.json result files.

Flattens nested labels/scores dicts into flat columns.
Excludes *_extras.json (partial duplicates of *_extra.json).

Output columns (mirrors level1_master.csv structure + L2-specific fields):
  Meta:    jd_id, cv, prompt, run_id, source, model, temperature
  Labels:  skills, role_relevance, domain_relevance, education, holistic
  Derived: confident (True/False), holistic_score (0–3),
           fit_score_100 = round((0.6×skill + 0.3×(0.4×role + 0.6×domain) + 0.1×edu) × 100, 1)
  Scores:  skill_score, role_score, domain_score, edu_score
  Human:   human_holistic_label
  Perf:    latency_s, prompt_tokens, output_tokens, total_tokens, error

Usage:
    python3 build_level2_master.py
"""

import json
import csv
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / 'results'

# files to include — exclude *_extras.json (subset of *_extra.json)
JSON_FILES = sorted(
    f for f in RESULTS.glob('level2_p0_*.json')
    if not f.name.endswith('_extras.json')
)

COLUMNS = [
    'jd_id', 'cv', 'prompt', 'run_id', 'source', 'model', 'temperature',
    'skills', 'role_relevance', 'domain_relevance', 'education', 'holistic',
    'confident', 'holistic_score', 'fit_score_100',
    'skill_score', 'role_score', 'domain_score', 'edu_score',
    'human_holistic_label',
    'latency_s', 'prompt_tokens', 'output_tokens', 'total_tokens', 'error',
]

HOLISTIC_MAP = {'STRONG': 3, 'MODERATE': 2, 'WEAK': 1, 'NO FIT': 0}


def fit_score_100(skill, role, domain, edu):
    if any(v is None for v in (skill, role, domain, edu)):
        return None
    return round((0.6 * skill + 0.3 * (0.4 * role + 0.6 * domain) + 0.1 * edu) * 100, 1)

OUT_PATH = RESULTS / 'level2_master.csv'


def flatten(record: dict) -> dict:
    labels = record.get('labels') or {}
    scores = record.get('scores') or {}

    skill  = scores.get('skill_score')
    role   = scores.get('role_score')
    domain = scores.get('domain_score')
    edu    = scores.get('edu_score')
    conf   = (labels.get('confidence') or '').upper()
    hol    = (labels.get('holistic') or '').upper()

    return {
        'jd_id':               record.get('jd_id'),
        'cv':                  record.get('cv'),
        'prompt':              record.get('prompt'),
        'run_id':              record.get('run_id'),
        'source':              record.get('source'),
        'model':               record.get('model'),
        'temperature':         record.get('temperature'),
        'skills':              labels.get('skills'),
        'role_relevance':      labels.get('role_relevance'),
        'domain_relevance':    labels.get('domain_relevance'),
        'education':           labels.get('education'),
        'holistic':            labels.get('holistic'),
        'confident':           True if conf == 'HIGH' else (False if conf == 'LOW' else None),
        'holistic_score':      HOLISTIC_MAP.get(hol),
        'fit_score_100':       fit_score_100(skill, role, domain, edu),
        'skill_score':         skill,
        'role_score':          role,
        'domain_score':        domain,
        'edu_score':           edu,
        'human_holistic_label': record.get('human_holistic_label'),
        'latency_s':           record.get('latency_s'),
        'prompt_tokens':       record.get('prompt_tokens'),
        'output_tokens':       record.get('output_tokens'),
        'total_tokens':        record.get('total_tokens'),
        'error':               record.get('error'),
    }


def main():
    rows = []
    for path in JSON_FILES:
        records = json.loads(path.read_text())
        for r in records:
            rows.append(flatten(r))
        print(f'  {path.name:<50} {len(records):>4} rows')

    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    valid = sum(1 for r in rows if r['error'] is None)
    print(f'\nTotal rows:  {len(rows)}')
    print(f'Valid calls: {valid}')
    print(f'Errors:      {len(rows) - valid}')

    import pandas as pd
    df = pd.read_csv(OUT_PATH)
    print('\nBreakdown by source × cv (run_id==1):')
    one = df[df['run_id'] == 1]
    print(one.groupby(['source', 'cv']).size().unstack(fill_value=0).to_string())
    print(f'\nSaved → {OUT_PATH}')


if __name__ == '__main__':
    main()
