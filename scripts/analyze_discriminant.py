#!/usr/bin/env python3
"""
Level 1 & 2 — discriminant validity (3-way flip test).

Compares three JD pools to test whether different CVs receive systematically different
scores on the same JDs.

SAMPLING NOTE:
- main: filtered to include only JDs where cv_primary was rated 2 or 3 by human
  (Consider / Apply verdicts). This removes deliberately-irrelevant JDs used for
  robustness testing.
- hr_extra: 18 JDs selected by keyword matching only (no human labels).
  May contain variable relevance to cv_hr.
- engineer_extra: 25 JDs selected by keyword matching only (no human labels).
  May contain variable relevance to cv_engineer.

Tests per pool:
  - Friedman omnibus: χ² + Kendall's W (effect size)
  - Pairwise Wilcoxon signed-rank + rank-biserial r (effect size)
  - RM-ANOVA (sensitivity check on continuous scales)
  - Descriptive means with bootstrap 95% CI

Output: STDOUT + results/analysis/level1_discriminant.csv (if updating)
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path
from scipy import stats
from itertools import combinations

ROOT       = Path(__file__).parent.parent
MASTER_L1  = ROOT / 'results' / 'level1_master.csv'
MASTER_L2  = ROOT / 'results' / 'level2_master.csv'
OUT_DIR    = ROOT / 'results' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = ['P0', 'P1', 'P2']
CVS     = ['cv_primary', 'cv_hr', 'cv_engineer']
N_BOOT  = 2000
RNG     = 42

# Threshold for filtering main: keep JDs where cv_primary human_holistic_label >= threshold
MAIN_FILTER_THRESHOLD = 2


def interpret_w(v):
    if pd.isna(v): return ''
    if v < 0.1: return 'negligible'
    if v < 0.3: return 'small'
    if v < 0.5: return 'medium'
    return 'large'


def interpret_rbc(v):
    if pd.isna(v): return ''
    a = abs(v)
    if a < 0.1: return 'negligible'
    if a < 0.3: return 'small'
    if a < 0.5: return 'medium'
    return 'large'


def aggregate_runs(df, score_col='score'):
    return df.groupby(['jd_id', 'cv', 'prompt'])[score_col].median().reset_index()


def pivot_wide(agg, prompt, score_col='score'):
    sub = agg[agg['prompt'] == prompt]
    wide = sub.pivot(index='jd_id', columns='cv', values=score_col).dropna()
    cols = [c for c in CVS if c in wide.columns]
    return wide[cols]


def kendalls_w(chi2, n, k):
    return chi2 / (n * (k - 1)) if n > 0 and k > 1 else np.nan


def rank_biserial_wilcoxon(x, y):
    diff = np.asarray(x) - np.asarray(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    sum_pos = ranks[diff > 0].sum()
    sum_neg = ranks[diff < 0].sum()
    total = sum_pos + sum_neg
    return float((sum_pos - sum_neg) / total) if total > 0 else 0.0


def bootstrap_ci_mean(values, n_boot=N_BOOT):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(RNG)
    means = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze_level1(df, subset_label, prompts=PROMPTS, score_col='score', n_contrasts_in_family=None):
    rows = []
    agg = aggregate_runs(df, score_col=score_col)
    scale = 'score_0-100' if score_col == 'score_100' else 'score_0-3'

    for prompt in prompts:
        wide = pivot_wide(agg, prompt, score_col=score_col)
        n, k = wide.shape
        cvs_present = list(wide.columns)
        if n < 5 or k < 2:
            continue

        # RM-ANOVA only for 0–100
        if score_col == 'score_100':
            long = wide.reset_index().melt(id_vars='jd_id', var_name='cv', value_name='score')
            aov = pg.rm_anova(data=long, dv='score', within='cv', subject='jd_id',
                              detailed=True, correction=True)
            r = aov.iloc[0]
            rows.append({
                'level': 'L1', 'subset': subset_label, 'prompt': prompt, 'scale': scale,
                'comparison': 'all CVs',
                'test': 'RM-ANOVA', 'n': n, 'k': k,
                'statistic':   round(float(r['F']), 3),
                'p_value':     round(float(r.get('p_GG_corr', r['p_unc'])), 6),
                'effect_size': round(float(r['ng2']), 3),
                'effect_name': 'partial η²',
                'interpretation': ('large'  if r['ng2'] >= 0.14 else
                                   'medium' if r['ng2'] >= 0.06 else
                                   'small'  if r['ng2'] >= 0.01 else 'negligible'),
            })
            continue

        # Friedman
        try:
            chi2, p_fri = stats.friedmanchisquare(*[wide[c].values for c in cvs_present])
            w = kendalls_w(chi2, n, k)
            rows.append({
                'level': 'L1', 'subset': subset_label, 'prompt': prompt,
                'comparison': 'all CVs',
                'test': 'Friedman', 'n': n, 'k': k,
                'statistic': round(float(chi2), 3),
                'p_value':   round(float(p_fri), 4),
                'effect_size': round(float(w), 3),
                'effect_name': "Kendall's W",
                'interpretation': interpret_w(w),
            })
        except Exception as e:
            rows.append({'level': 'L1', 'subset': subset_label, 'prompt': prompt,
                         'comparison': 'all CVs', 'test': 'Friedman', 'error': str(e)})

        # Pairwise Wilcoxon
        pairs = list(combinations(cvs_present, 2))
        denom = n_contrasts_in_family if n_contrasts_in_family else len(pairs)
        alpha_bon = 0.05 / denom
        for a, b in pairs:
            x, y = wide[a].values, wide[b].values
            try:
                w_stat, p_w = stats.wilcoxon(x, y)
            except ValueError:
                w_stat, p_w = 0.0, 1.0
            rbc = rank_biserial_wilcoxon(x, y)
            mean_diff = float(np.mean(x - y))
            sig = '*' if p_w < alpha_bon else ''
            rows.append({
                'level': 'L1', 'subset': subset_label, 'prompt': prompt,
                'comparison': f'{a} − {b}',
                'test': 'Wilcoxon', 'n': n, 'k': 2,
                'statistic': round(float(w_stat), 3),
                'p_value':   round(float(p_w), 4),
                'effect_size': round(rbc, 3),
                'effect_name': 'Rank-biserial r',
                'mean_diff':   round(mean_diff, 3),
                'sig_after_bonferroni': sig,
                'alpha_bonferroni': round(alpha_bon, 4),
                'interpretation': interpret_rbc(rbc),
            })

        # Descriptive
        for cv in cvs_present:
            vals = wide[cv].values
            lo, hi = bootstrap_ci_mean(vals)
            rows.append({
                'level': 'L1', 'subset': subset_label, 'prompt': prompt,
                'comparison': cv,
                'test': 'Descriptive mean', 'n': n, 'k': 1,
                'statistic': round(float(np.mean(vals)), 3),
                'ci_low': round(lo, 3), 'ci_high': round(hi, 3),
                'median': round(float(np.median(vals)), 3),
            })

    return rows


def analyze_level2(df, subset_label, score_col, n_contrasts_in_family=None):
    rows = []
    agg = df.groupby(['jd_id', 'cv'])[score_col].median().reset_index()
    wide = agg.pivot(index='jd_id', columns='cv', values=score_col).dropna()
    cols = [c for c in CVS if c in wide.columns]
    wide = wide[cols]
    n, k = wide.shape
    cvs_present = list(wide.columns)
    if n < 5 or k < 2:
        return rows

    # Friedman
    chi2, p_fri = stats.friedmanchisquare(*[wide[c].values for c in cvs_present])
    w = kendalls_w(chi2, n, k)
    rows.append({
        'level': 'L2', 'subset': subset_label, 'score_col': score_col,
        'comparison': 'all CVs',
        'test': 'Friedman', 'n': n, 'k': k,
        'statistic': round(float(chi2), 3),
        'p_value':   round(float(p_fri), 4),
        'effect_size': round(float(w), 3),
        'effect_name': "Kendall's W",
        'interpretation': interpret_w(w),
    })

    # Pairwise Wilcoxon
    pairs = list(combinations(cvs_present, 2))
    denom = n_contrasts_in_family if n_contrasts_in_family else len(pairs)
    alpha_bon = 0.05 / denom
    for a, b in pairs:
        x, y = wide[a].values, wide[b].values
        try:
            w_stat, p_w = stats.wilcoxon(x, y)
        except ValueError:
            w_stat, p_w = 0.0, 1.0
        rbc = rank_biserial_wilcoxon(x, y)
        mean_diff = float(np.mean(x - y))
        sig = '*' if p_w < alpha_bon else ''
        rows.append({
            'level': 'L2', 'subset': subset_label, 'score_col': score_col,
            'comparison': f'{a} − {b}',
            'test': 'Wilcoxon', 'n': n, 'k': 2,
            'statistic': round(float(w_stat), 3),
            'p_value':   round(float(p_w), 4),
            'effect_size': round(rbc, 3),
            'effect_name': 'Rank-biserial r',
            'mean_diff':   round(mean_diff, 3),
            'sig_after_bonferroni': sig,
            'alpha_bonferroni': round(alpha_bon, 4),
            'interpretation': interpret_rbc(rbc),
        })

    # RM-ANOVA
    long = wide.reset_index().melt(id_vars='jd_id', var_name='cv', value_name='score')
    aov = pg.rm_anova(data=long, dv='score', within='cv', subject='jd_id',
                      detailed=True, correction=True)
    r = aov.iloc[0]
    rows.append({
        'level': 'L2', 'subset': subset_label, 'score_col': score_col,
        'comparison': 'all CVs',
        'test': 'RM-ANOVA', 'n': n, 'k': k,
        'statistic':   round(float(r['F']), 3),
        'p_value':     round(float(r.get('p_GG_corr', r['p_unc'])), 6),
        'effect_size': round(float(r['ng2']), 3),
        'effect_name': 'partial η²',
        'interpretation': ('large'  if r['ng2'] >= 0.14 else
                           'medium' if r['ng2'] >= 0.06 else
                           'small'  if r['ng2'] >= 0.01 else 'negligible'),
    })

    # Descriptive
    for cv in cvs_present:
        vals = wide[cv].values
        lo, hi = bootstrap_ci_mean(vals)
        rows.append({
            'level': 'L2', 'subset': subset_label, 'score_col': score_col,
            'comparison': cv,
            'test': 'Descriptive mean', 'n': n, 'k': 1,
            'statistic': round(float(np.mean(vals)), 3),
            'ci_low': round(lo, 3), 'ci_high': round(hi, 3),
            'median': round(float(np.median(vals)), 3),
        })

    return rows


def main():
    l1 = pd.read_csv(MASTER_L1)
    l2 = pd.read_csv(MASTER_L2)

    print('=' * 110)
    print('DISCRIMINANT VALIDITY — 3-way flip test')
    print('=' * 110)

    # ── Filter main for L1 & L2 ───────────────────────────────────────────────
    jds_main = l1[
        (l1['source'] != 'engineer_extra') &
        (l1['cv'] == 'cv_primary') &
        (l1['human_holistic_label'].fillna(0) >= MAIN_FILTER_THRESHOLD)
    ]['jd_id'].unique()

    main_l1 = l1[l1['jd_id'].isin(jds_main)].copy()
    main_l2 = l2[l2['jd_id'].isin(jds_main)].copy()

    hr_l1 = l1[l1['source'] == 'hr_extra'].copy()
    hr_l2 = l2[l2['source'] == 'hr_extra'].copy()

    eng_l1 = l1[l1['source'] == 'engineer_extra'].copy()
    eng_l2 = l2[l2['source'] == 'engineer_extra'].copy()

    n_main = main_l1[main_l1['cv'] == 'cv_primary']['jd_id'].nunique()
    n_hr = hr_l1[hr_l1['cv'] == 'cv_primary']['jd_id'].nunique()
    n_eng = eng_l1[eng_l1['cv'] == 'cv_primary']['jd_id'].nunique()

    print(f'\nSample sizes:')
    print(f'  main:           {n_main} JDs (human_holistic_label >= {MAIN_FILTER_THRESHOLD})')
    print(f'  hr_extra:       {n_hr} JDs (keyword-selected, no human labels)')
    print(f'  engineer_extra: {n_eng} JDs (keyword-selected, no human labels)')
    print()

    all_rows = []

    # ── Level 1 ────────────────────────────────────────────────────────────────
    all_rows += analyze_level1(main_l1, 'main', n_contrasts_in_family=9)
    p2_main = main_l1[main_l1['prompt'] == 'P2']
    all_rows += analyze_level1(p2_main, 'main', prompts=['P2'], score_col='score_100')

    all_rows += analyze_level1(hr_l1, 'hr_extra', prompts=['P2'], n_contrasts_in_family=3)
    all_rows += analyze_level1(hr_l1, 'hr_extra', prompts=['P2'], score_col='score_100')

    all_rows += analyze_level1(eng_l1, 'engineer_extra', prompts=['P2'], n_contrasts_in_family=3)
    all_rows += analyze_level1(eng_l1, 'engineer_extra', prompts=['P2'], score_col='score_100')

    # ── Level 2 ────────────────────────────────────────────────────────────────
    for score_col in ['holistic_score', 'fit_score_100']:
        all_rows += analyze_level2(main_l2, 'main', score_col, n_contrasts_in_family=6)
        all_rows += analyze_level2(hr_l2, 'hr_extra', score_col, n_contrasts_in_family=6)
        all_rows += analyze_level2(eng_l2, 'engineer_extra', score_col, n_contrasts_in_family=6)

    out = pd.DataFrame(all_rows)

    # ── Pretty print ──────────────────────────────────────────────────────────
    def section(title, mask, cols):
        print('\n' + '=' * 110)
        print(title)
        print('=' * 110)
        view = out[mask]
        existing = [c for c in cols if c in view.columns]
        print(view[existing].to_string(index=False))

    section(
        'Friedman omnibus (Kendall W)',
        out['test'] == 'Friedman',
        ['level', 'subset', 'prompt', 'score_col', 'n', 'statistic', 'p_value', 'effect_size', 'interpretation'],
    )
    section(
        'Pairwise Wilcoxon (α Bonferroni-corrected; * = significant)',
        out['test'] == 'Wilcoxon',
        ['level', 'subset', 'prompt', 'score_col', 'comparison', 'mean_diff', 'p_value',
         'effect_size', 'sig_after_bonferroni', 'interpretation'],
    )
    section(
        'RM-ANOVA sensitivity (Greenhouse-Geisser corrected)',
        out['test'] == 'RM-ANOVA',
        ['level', 'subset', 'score_col', 'n', 'statistic', 'p_value', 'effect_size', 'interpretation'],
    )
    section(
        'Descriptive — mean per CV (bootstrap 95% CI)',
        out['test'] == 'Descriptive mean',
        ['level', 'subset', 'prompt', 'score_col', 'comparison', 'statistic', 'ci_low', 'ci_high', 'median'],
    )

    print(f'\n(No CSV saved; output to STDOUT only)\n')


if __name__ == '__main__':
    main()
