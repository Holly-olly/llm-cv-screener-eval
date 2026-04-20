#!/usr/bin/env python3
"""
Track B — Prepare dataset for predictive validity analysis.

Source: evaluation/scripts/migration-output.json (68 historical jobs with real outcomes)

Outcome coding:
  outcome = 1 (positive — got to interview/shortlist):
      🤝 Interview, 📉 Rejected (Int), 🎯 Target
  outcome = 0 (negative — not progressed):
      ❌ Rejected (App), 🚫 Dismissed
  outcome = None (unknown — applied, no result yet):
      📨 Applied, 👀 New

Selection logic:
  - All positives (N is small, keep every one)
  - Up to 12 negatives with jd length ≥ 700 chars (prefer variety in score range)
  - Up to 7 unknowns with jd length ≥ 700 chars (labelled separately in output)
  - Dedup by (company_lower, role_lower[:20]) — removes identical IQVIA duplicate
  - Exclude entries with jd_len < 300 chars (too short to be a real JD)

Output: evaluation/data/batch-results/track_b_prepared.json
  Fields per entry: id, company, role, jd, old_ai_score, old_status, outcome, outcome_label

0 API calls — pure data curation.

Usage:
    cd /path/to/cool-cohen
    python3 evaluation/track_b_prepare.py
"""

import json
import random
from pathlib import Path

RANDOM_SEED = 42

# Outcome mapping
POSITIVE_STATUSES = {'🤝 Interview', '📉 Rejected (Int)', '🎯 Target'}
NEGATIVE_STATUSES  = {'❌ Rejected (App)', '🚫 Dismissed'}
UNKNOWN_STATUSES   = {'📨 Applied', '👀 New'}

MIN_JD_LEN   = 300   # chars — exclude near-empty JD fields
NEG_SAMPLE   = 12    # how many negatives to include
UNK_SAMPLE   = 7     # how many unknowns to include


def outcome_label(status: str) -> str:
    if status in POSITIVE_STATUSES: return 'interview'
    if status in NEGATIVE_STATUSES: return 'rejection'
    return 'unknown'


def dedup(entries):
    """Remove exact duplicates by (company_lower, role_prefix_lower)."""
    seen = set()
    result = []
    for e in entries:
        key = (e['company'].strip().lower(), (e['role'] or '')[:20].lower())
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


def main():
    src = Path('evaluation/scripts/migration-output.json')
    out = Path('evaluation/data/batch-results/track_b_prepared.json')

    raw = json.loads(src.read_text())
    print(f'Loaded {len(raw)} entries from migration-output.json')

    # Filter: require non-empty JD
    raw = [d for d in raw if len(d.get('jd', '')) >= MIN_JD_LEN]
    print(f'After jd_len >= {MIN_JD_LEN} filter: {len(raw)} entries')

    # Split by outcome
    positives = dedup([d for d in raw if d['status'] in POSITIVE_STATUSES])
    negatives  = dedup([d for d in raw if d['status'] in NEGATIVE_STATUSES])
    unknowns   = dedup([d for d in raw if d['status'] in UNKNOWN_STATUSES])

    print(f'\nOutcome groups (after dedup):')
    print(f'  Positives (interview/target): {len(positives)}')
    print(f'  Negatives (rejected/dismissed): {len(negatives)}')
    print(f'  Unknown (applied, no result): {len(unknowns)}')

    # Sample negatives: spread across score range for variety
    rng = random.Random(RANDOM_SEED)
    negatives_sorted = sorted(negatives, key=lambda d: d.get('score') or 0)
    if len(negatives_sorted) > NEG_SAMPLE:
        # Stratified: pick evenly across sorted order
        step = len(negatives_sorted) / NEG_SAMPLE
        neg_sample = [negatives_sorted[int(i * step)] for i in range(NEG_SAMPLE)]
    else:
        neg_sample = negatives_sorted

    # Sample unknowns
    unk_sample = rng.sample(unknowns, min(UNK_SAMPLE, len(unknowns)))

    # Combine: all positives + sampled negatives + sampled unknowns
    selected = positives + neg_sample + unk_sample

    # Build output records
    records = []
    for d in selected:
        outcome = 1 if d['status'] in POSITIVE_STATUSES else (
                  0 if d['status'] in NEGATIVE_STATUSES else None)
        records.append({
            'id':           d['id'],
            'company':      d['company'].strip(),
            'role':         (d['role'] or '').strip(),
            'jd':           d['jd'],
            'jd_len':       len(d['jd']),
            'old_ai_score': d.get('score'),
            'old_status':   d['status'],
            'outcome':      outcome,
            'outcome_label': outcome_label(d['status']),
        })

    # Sort for readability: positives first, then negatives, then unknowns
    order = {'interview': 0, 'rejection': 1, 'unknown': 2}
    records.sort(key=lambda r: (order[r['outcome_label']], -(r['old_ai_score'] or 0)))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    # Summary table
    print(f'\n{"Company":<30} {"Role"[:30]:<30} {"Old score":>9} {"Status":<25} {"Outcome"}')
    print('-' * 105)
    for r in records:
        print(f'{r["company"][:29]:<30} {r["role"][:29]:<30} {str(r["old_ai_score"] or "?"):>9} '
              f'{r["old_status"]:<25} {r["outcome_label"]}')

    pos_count = sum(1 for r in records if r['outcome'] == 1)
    neg_count = sum(1 for r in records if r['outcome'] == 0)
    unk_count = sum(1 for r in records if r['outcome'] is None)

    print(f'\nFinal dataset: {len(records)} entries total')
    print(f'  Positive (outcome=1): {pos_count}')
    print(f'  Negative (outcome=0): {neg_count}')
    print(f'  Unknown  (outcome=None): {unk_count}')
    print(f'\nOld AI score stats (positive group): '
          f'mean={sum(r["old_ai_score"] or 0 for r in records if r["outcome"]==1)/max(pos_count,1):.1f}')
    print(f'Old AI score stats (negative group): '
          f'mean={sum(r["old_ai_score"] or 0 for r in records if r["outcome"]==0)/max(neg_count,1):.1f}')
    print(f'\nSaved to: {out}')


if __name__ == '__main__':
    main()
