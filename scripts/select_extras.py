#!/usr/bin/env python3
"""
Add extra JDs to the Level 1 evaluation set.

The original 32 labeled JDs lean toward psychometric / assessment roles, so
the LLM was getting too few signals on cv_hr and cv_engineer. To strengthen
those CVs we add JDs from the unlabeled pool, filtered by role keywords.

How it works:
  1. Scan data/unlabeled-jds/ for filenames containing the pool's keywords
  2. Score each match with the L1 P2 prompt for the chosen CV (3 runs each)
  3. Save to results/level1_p2_{cv}_extra_{pool}.json
     with source = "{pool}_extra" so it stays distinguishable from main_32

Usage:
    python3 select_extras.py engineer cv_engineer
    python3 select_extras.py hr cv_hr
    python3 select_extras.py hr cv_primary --runs 3
"""

import json
import re
import sys
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import google.generativeai as genai

sys.path.insert(0, str(Path(__file__).parent))
from score_level1_p2 import score_one, load_api_key, GEMINI_MODEL  # noqa: F401

ROOT    = Path(__file__).parent.parent
DATA    = ROOT / 'data'
RESULTS = ROOT / 'results'

KEYWORDS = {
    'engineer': ['software', 'system', 'solution', 'architect'],
    'hr':       ['people', 'hr'],
}


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit('Usage: select_extras.py POOL CV [--runs N]\n'
                 '  POOL = engineer | hr')
    pool, cv_name = args[0], args[1]
    runs = int(args[args.index('--runs') + 1]) if '--runs' in args else 3

    if pool not in KEYWORDS:
        sys.exit(f'Unknown pool "{pool}". Choose: {list(KEYWORDS)}')

    # 1. Find matching JDs in unlabeled folder
    pattern = r'\b(?:' + '|'.join(KEYWORDS[pool]) + r')\b'
    jds = sorted(p for p in (DATA / 'unlabeled-jds').glob('*.txt')
                 if re.search(pattern, p.stem.lower()))
    print(f'Pool "{pool}" — {len(jds)} JDs match keywords {KEYWORDS[pool]}:')
    for p in jds:
        print(f'  {p.stem}')
    print()

    # 2. Score each JD against the chosen CV with the L1 P2 prompt
    cv_text = (DATA / 'cv' / f'{cv_name}.txt').read_text()
    genai.configure(api_key=load_api_key())
    client = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config=genai.GenerationConfig(temperature=1.0),
    )
    print(f'Scoring against {cv_name} with P2 prompt, {runs} runs each…\n')

    results = []
    for path in jds:
        jd_id   = path.stem.strip()
        jd_text = path.read_text()
        for run_id in range(1, runs + 1):
            r = score_one(client, cv_text, jd_text, jd_id)
            r.update({
                'run_id':               run_id,
                'cv':                   cv_name,
                'prompt':               'P2',
                'model':                GEMINI_MODEL,
                'temperature':          1.0,
                'source':               f'{pool}_extra',
                'human_holistic_label': None,
            })
            results.append(r)
            print(f'  {jd_id[:55]:<55} [{run_id}/{runs}]  score={r["score"]}  verdict={r["verdict"]}')
            time.sleep(2)

    out = RESULTS / f'level1_p2_{cv_name}_extra_{pool}.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f'\nSaved {len(results)} records to {out.name}')


if __name__ == '__main__':
    main()
