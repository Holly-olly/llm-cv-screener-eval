#!/usr/bin/env python3
"""
Level 1 — reliability analysis (ICC + verdict agreement).

Reads results/level1_master.csv. Computes:
  1) Score 0–3   ICC by prompt (CVs pooled)              — 3 numbers
  2) Score 0–3   ICC by (prompt × cv)                    — 9 numbers
  3) Score 0–100 ICC for P2 (pooled + per cv)            — 4 numbers
  4) Verdict     Fleiss κ + %unanimous + %majority,
                 by prompt (pooled) and (prompt × cv)    — 12 numbers

ICC type: two-way random, average measures = ICC(A,k) in pingouin.
SEM = SD × √(1 − ICC).
"Subject" when pooling across CVs = unique (cv, jd_id) pair.

Output: results/analysis/level1_icc.csv  + printed tables.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path
from statsmodels.stats.inter_rater import fleiss_kappa

ROOT       = Path(__file__).parent.parent
MASTER_CSV = ROOT / 'results' / 'level1_master.csv'
OUT_DIR    = ROOT / 'results' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV    = OUT_DIR / 'level1_icc.csv'

PROMPTS = ['P0', 'P1', 'P2']
CVS     = ['cv_primary', 'cv_hr', 'cv_engineer']
VERDICT_ORDER = ['Skip', 'Consider', 'Apply']


def interpret_icc(v):
    if pd.isna(v): return ''
    if v < 0.50: return 'poor'
    if v < 0.75: return 'moderate'
    if v < 0.90: return 'good'
    return 'excellent'


def interpret_kappa(v):
    if pd.isna(v): return ''
    if v < 0.20: return 'slight'
    if v < 0.40: return 'fair'
    if v < 0.60: return 'moderate'
    if v < 0.80: return 'substantial'
    return 'almost perfect'


def compute_icc(sub, target_col, score_col):
    sub = sub[sub[score_col].notna()].copy()
    if sub.empty or sub[target_col].nunique() < 2:
        return None
    sub[score_col] = sub[score_col].astype(float)

    icc = pg.intraclass_corr(
        data=sub, targets=target_col, raters='run_id', ratings=score_col,
    )
    row_k = icc[icc['Type'] == 'ICC(A,k)'].iloc[0]   # average-measure (reported)
    row_1 = icc[icc['Type'] == 'ICC(A,1)'].iloc[0]   # single-measure (operationally relevant)

    sd  = sub[score_col].std()
    val_k = float(row_k['ICC']); ci_k = row_k['CI95']
    val_1 = float(row_1['ICC']); ci_1 = row_1['CI95']
    sem   = sd * np.sqrt(max(0.0, 1 - val_k))

    return {
        'n_subjects':     sub[target_col].nunique(),
        'n_raters':       sub['run_id'].nunique(),
        'value':          round(val_k, 3),
        'ci_low':         round(float(ci_k[0]), 3),
        'ci_high':        round(float(ci_k[1]), 3),
        'icc_1':          round(val_1, 3),
        'icc_1_ci_low':   round(float(ci_1[0]), 3),
        'icc_1_ci_high': round(float(ci_1[1]), 3),
        'sem':            round(sem, 3),
        'sd':             round(sd, 3),
        'interpretation': interpret_icc(val_k),
    }


def compute_verdict(sub, target_col):
    sub = sub[sub['verdict'].notna()].copy()
    sub = sub[sub['verdict'].astype(str).str.strip().ne('')]
    if sub.empty:
        return None

    wide = sub.pivot_table(index=target_col, columns='run_id',
                           values='verdict', aggfunc='first').dropna()
    if wide.empty:
        return None

    n_subj, n_rat = wide.shape
    table = np.zeros((n_subj, len(VERDICT_ORDER)), dtype=int)
    for i, (_, vals) in enumerate(wide.iterrows()):
        for v in vals.values:
            if v in VERDICT_ORDER:
                table[i, VERDICT_ORDER.index(v)] += 1

    try:
        k = float(fleiss_kappa(table))
    except Exception:
        k = np.nan

    unanimous = wide.apply(lambda r: r.nunique() == 1, axis=1).mean() * 100
    majority  = wide.apply(
        lambda r: max(list(r.values).count(v) for v in set(r.values)) >= 2,
        axis=1,
    ).mean() * 100

    return {
        'n_subjects':     n_subj,
        'n_raters':       n_rat,
        'value':          round(k, 3),
        'pct_unanimous':  round(unanimous, 1),
        'pct_majority':   round(majority, 1),
        'interpretation': interpret_kappa(k),
    }


def main():
    df = pd.read_csv(MASTER_CSV)
    df['subject_pooled'] = df['cv'] + '|' + df['jd_id']
    print(f'Loaded {len(df)} rows from {MASTER_CSV.name}')

    rows = []

    # 1) Score 0–3, pooled by prompt
    for p in PROMPTS:
        r = compute_icc(df[df['prompt'] == p], 'subject_pooled', 'score')
        if r:
            rows.append({'scenario': 'score_0-3 pooled', 'prompt': p, 'cv': 'ALL',
                         'metric': 'ICC(A,k)', **r,
                         'pct_unanimous': '', 'pct_majority': ''})

    # 2) Score 0–3, per (prompt × cv)
    for p in PROMPTS:
        for cv in CVS:
            sub = df[(df['prompt'] == p) & (df['cv'] == cv)]
            r = compute_icc(sub, 'jd_id', 'score')
            if r:
                rows.append({'scenario': 'score_0-3 per_cv', 'prompt': p, 'cv': cv,
                             'metric': 'ICC(A,k)', **r,
                             'pct_unanimous': '', 'pct_majority': ''})

    # 3) Score 0–100, P2 only
    p2 = df[df['prompt'] == 'P2']
    r = compute_icc(p2, 'subject_pooled', 'score_100')
    if r:
        rows.append({'scenario': 'score_0-100 pooled', 'prompt': 'P2', 'cv': 'ALL',
                     'metric': 'ICC(A,k)', **r,
                     'pct_unanimous': '', 'pct_majority': ''})
    for cv in CVS:
        r = compute_icc(p2[p2['cv'] == cv], 'jd_id', 'score_100')
        if r:
            rows.append({'scenario': 'score_0-100 per_cv', 'prompt': 'P2', 'cv': cv,
                         'metric': 'ICC(A,k)', **r,
                         'pct_unanimous': '', 'pct_majority': ''})

    # 4) Verdict — Fleiss κ + %unanimous + %majority
    for p in PROMPTS:
        r = compute_verdict(df[df['prompt'] == p], 'subject_pooled')
        if r:
            rows.append({'scenario': 'verdict pooled', 'prompt': p, 'cv': 'ALL',
                         'metric': 'Fleiss κ', **r,
                         'ci_low': '', 'ci_high': '', 'sem': '', 'sd': ''})
    for p in PROMPTS:
        for cv in CVS:
            sub = df[(df['prompt'] == p) & (df['cv'] == cv)]
            r = compute_verdict(sub, 'jd_id')
            if r:
                rows.append({'scenario': 'verdict per_cv', 'prompt': p, 'cv': cv,
                             'metric': 'Fleiss κ', **r,
                             'ci_low': '', 'ci_high': '', 'sem': '', 'sd': ''})

    cols = ['scenario', 'prompt', 'cv', 'metric',
            'n_subjects', 'n_raters',
            'value', 'ci_low', 'ci_high',
            'icc_1', 'icc_1_ci_low', 'icc_1_ci_high',
            'sem', 'sd',
            'pct_unanimous', 'pct_majority',
            'interpretation']
    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = ''
    out = out[cols]
    out.to_csv(OUT_CSV, index=False)

    # ── Pretty print ──────────────────────────────────────────────────────────
    def print_section(title, mask, columns):
        print('\n' + '=' * 100)
        print(title)
        print('=' * 100)
        print(out[mask][columns].to_string(index=False))

    print_section(
        'Score 0–3 — ICC(A,k), two-way random, average measures',
        out['scenario'].str.startswith('score_0-3'),
        ['scenario', 'prompt', 'cv', 'n_subjects',
         'value', 'ci_low', 'ci_high', 'sem', 'sd', 'interpretation'],
    )
    print_section(
        'Score 0–100 (P2) — ICC(A,k)',
        out['scenario'].str.startswith('score_0-100'),
        ['scenario', 'prompt', 'cv', 'n_subjects',
         'value', 'ci_low', 'ci_high', 'sem', 'sd', 'interpretation'],
    )
    print_section(
        'Verdict — Fleiss κ + agreement %',
        out['metric'] == 'Fleiss κ',
        ['scenario', 'prompt', 'cv', 'n_subjects',
         'value', 'pct_unanimous', 'pct_majority', 'interpretation'],
    )

    print(f'\nSaved: {OUT_CSV}')


if __name__ == '__main__':
    main()
