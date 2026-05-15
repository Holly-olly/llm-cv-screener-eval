#!/usr/bin/env python3
"""
Level 2 — reliability analysis (ICC + holistic-label agreement).

Reads results/level2_master.csv. Only one prompt (L2_P0), so all output
is sliced by CV (+ pooled across CVs).

Computes:
  1) holistic_score 0–3   ICC(A,k)        — pooled + per CV
  2) holistic label       Fleiss κ        — pooled + per CV
                          (STRONG / MODERATE / WEAK / NO FIT)
  3) fit_score_100        ICC(A,k)        — pooled + per CV

Subscales (skill_score, role_score, domain_score, edu_score) — deferred to
deeper analysis after holistic findings.

ICC type: two-way random, average measures = ICC(A,k) in pingouin.
SEM = SD × √(1 − ICC).
"Subject" when pooling across CVs = unique (cv, jd_id) pair.

Output: results/analysis/level2_icc.csv  + printed tables.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path
from statsmodels.stats.inter_rater import fleiss_kappa

ROOT       = Path(__file__).parent.parent
MASTER_CSV = ROOT / 'results' / 'level2_master.csv'
OUT_DIR    = ROOT / 'results' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV    = OUT_DIR / 'level2_icc.csv'

CVS              = ['cv_primary', 'cv_hr', 'cv_engineer']
HOLISTIC_LABELS  = ['NO FIT', 'WEAK', 'MODERATE', 'STRONG']


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

    # ICC requires balanced data: drop subjects that don't have all runs
    n_runs    = sub['run_id'].nunique()
    full_subj = sub.groupby(target_col)['run_id'].nunique()
    keep      = full_subj[full_subj == n_runs].index
    sub       = sub[sub[target_col].isin(keep)]
    if sub.empty or sub[target_col].nunique() < 2:
        return None

    icc = pg.intraclass_corr(
        data=sub, targets=target_col, raters='run_id', ratings=score_col,
    )
    row_k = icc[icc['Type'] == 'ICC(A,k)'].iloc[0]
    row_1 = icc[icc['Type'] == 'ICC(A,1)'].iloc[0]

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


def compute_kappa_label(sub, target_col, label_col, label_order):
    sub = sub[sub[label_col].notna()].copy()
    sub = sub[sub[label_col].astype(str).str.strip().ne('')]
    if sub.empty:
        return None

    wide = sub.pivot_table(index=target_col, columns='run_id',
                           values=label_col, aggfunc='first').dropna()
    if wide.empty:
        return None

    n_subj, n_rat = wide.shape
    table = np.zeros((n_subj, len(label_order)), dtype=int)
    for i, (_, vals) in enumerate(wide.iterrows()):
        for v in vals.values:
            if v in label_order:
                table[i, label_order.index(v)] += 1

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
    df = df[df['error'].isna()].copy()
    df['subject_pooled'] = df['cv'] + '|' + df['jd_id']
    print(f'Loaded {len(df)} valid rows from {MASTER_CSV.name}')
    print(f'Unique JDs: {df["jd_id"].nunique()}  CVs: {df["cv"].nunique()}  Runs: {sorted(df["run_id"].unique())}\n')

    rows = []

    # ── 1) holistic_score (0–3) — ICC ────────────────────────────────────────
    r = compute_icc(df, 'subject_pooled', 'holistic_score')
    if r:
        rows.append({'scenario': 'holistic_score_0-3 pooled', 'prompt': 'L2_P0', 'cv': 'ALL',
                     'metric': 'ICC(A,k)', **r,
                     'pct_unanimous': '', 'pct_majority': ''})
    for cv in CVS:
        r = compute_icc(df[df['cv'] == cv], 'jd_id', 'holistic_score')
        if r:
            rows.append({'scenario': 'holistic_score_0-3 per_cv', 'prompt': 'L2_P0', 'cv': cv,
                         'metric': 'ICC(A,k)', **r,
                         'pct_unanimous': '', 'pct_majority': ''})

    # ── 2) holistic label — Fleiss κ ─────────────────────────────────────────
    r = compute_kappa_label(df, 'subject_pooled', 'holistic', HOLISTIC_LABELS)
    if r:
        rows.append({'scenario': 'holistic_label pooled', 'prompt': 'L2_P0', 'cv': 'ALL',
                     'metric': 'Fleiss κ', **r,
                     'ci_low': '', 'ci_high': '', 'sem': '', 'sd': ''})
    for cv in CVS:
        r = compute_kappa_label(df[df['cv'] == cv], 'jd_id', 'holistic', HOLISTIC_LABELS)
        if r:
            rows.append({'scenario': 'holistic_label per_cv', 'prompt': 'L2_P0', 'cv': cv,
                         'metric': 'Fleiss κ', **r,
                         'ci_low': '', 'ci_high': '', 'sem': '', 'sd': ''})

    # ── 3) fit_score_100 — ICC ───────────────────────────────────────────────
    r = compute_icc(df, 'subject_pooled', 'fit_score_100')
    if r:
        rows.append({'scenario': 'fit_score_100 pooled', 'prompt': 'L2_P0', 'cv': 'ALL',
                     'metric': 'ICC(A,k)', **r,
                     'pct_unanimous': '', 'pct_majority': ''})
    for cv in CVS:
        r = compute_icc(df[df['cv'] == cv], 'jd_id', 'fit_score_100')
        if r:
            rows.append({'scenario': 'fit_score_100 per_cv', 'prompt': 'L2_P0', 'cv': cv,
                         'metric': 'ICC(A,k)', **r,
                         'pct_unanimous': '', 'pct_majority': ''})

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

    def print_section(title, mask, columns):
        print('\n' + '=' * 100)
        print(title)
        print('=' * 100)
        print(out[mask][columns].to_string(index=False))

    print_section(
        'Holistic score 0–3 — ICC(A,k)',
        out['scenario'].str.startswith('holistic_score_0-3'),
        ['scenario', 'cv', 'n_subjects',
         'value', 'ci_low', 'ci_high', 'sem', 'sd', 'interpretation'],
    )
    print_section(
        'Holistic label — Fleiss κ + agreement %',
        out['scenario'].str.startswith('holistic_label'),
        ['scenario', 'cv', 'n_subjects',
         'value', 'pct_unanimous', 'pct_majority', 'interpretation'],
    )
    print_section(
        'fit_score_100 — ICC(A,k)',
        out['scenario'].str.startswith('fit_score_100'),
        ['scenario', 'cv', 'n_subjects',
         'value', 'ci_low', 'ci_high', 'sem', 'sd', 'interpretation'],
    )

    print(f'\nSaved: {OUT_CSV}')


if __name__ == '__main__':
    main()
