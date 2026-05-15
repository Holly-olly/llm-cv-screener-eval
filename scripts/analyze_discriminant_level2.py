#!/usr/bin/env python3
"""
Level 2 — discriminant validity.

Tests whether different CVs receive systematically different LLM scores
on the same JDs (within-JD comparison of CVs).

Reads results/level2_master.csv. Aggregates 3 runs per (JD, CV) → median.

Three subsets × two score columns:
  subsets: main_32 (psych JDs), engineer_extra_25, hr_extra_18 — flip test on all
  scores : holistic_score (0–3 mapped from LLM holistic label)
           fit_score_100  (code-computed from skill/role/domain/edu weights)

For each (subset × score_col):
  - Friedman omnibus
  - Kendall's W
  - Pairwise Wilcoxon signed-rank (Bonferroni)
  - Rank-biserial r effect size
  - Cross-CV Spearman ρ (LOW = good discrimination)
  - Descriptive mean per CV (bootstrap 95% CI)

Output: results/analysis/level2_discriminant.csv + readable tables.
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
MASTER_CSV = ROOT / 'results' / 'level2_master.csv'
OUT_DIR    = ROOT / 'results' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV    = OUT_DIR / 'level2_discriminant.csv'

CVS         = ['cv_primary', 'cv_hr', 'cv_engineer']
SCORE_COLS  = ['holistic_score', 'fit_score_100']
SUBSET_MAP  = {
    'main_32':            'main',
    'engineer_extra_25':  'engineer_extra',
    'hr_extra_18':        'hr_extra',
}
N_BOOT  = 2000
RNG     = 42


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


def aggregate_runs(df, score_col):
    return df.groupby(['jd_id', 'cv'])[score_col].median().reset_index()


def pivot_wide(agg, score_col):
    wide = agg.pivot(index='jd_id', columns='cv', values=score_col).dropna()
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


def analyze(df, subset_label, score_col, n_contrasts_in_family=None):
    """
    n_contrasts_in_family: total Wilcoxon contrasts in this subset's family
    (across all score_cols). Used as Bonferroni denominator.
    """
    rows = []
    agg = aggregate_runs(df, score_col=score_col)
    wide = pivot_wide(agg, score_col)
    n, k = wide.shape
    cvs_present = list(wide.columns)
    if n < 5 or k < 2:
        return rows

    # ── Friedman omnibus ─────────────────────────────────────────────────────
    chi2, p_fri = stats.friedmanchisquare(*[wide[c].values for c in cvs_present])
    w = kendalls_w(chi2, n, k)
    rows.append({
        'subset': subset_label, 'score_col': score_col,
        'comparison': 'all CVs',
        'test': 'Friedman', 'n': n, 'k': k,
        'statistic': round(float(chi2), 3),
        'p_value':   round(float(p_fri), 4),
        'effect_size': round(float(w), 3),
        'effect_name': "Kendall's W",
        'mean_diff':   '',
        'interpretation': interpret_w(w),
    })

    # ── Pairwise Wilcoxon ────────────────────────────────────────────────────
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
            'subset': subset_label, 'score_col': score_col,
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

    # ── RM-ANOVA sensitivity (only for continuous fit_score_100) ─────────────
    if score_col == 'fit_score_100':
        long = wide.reset_index().melt(id_vars='jd_id', var_name='cv', value_name='score')
        aov = pg.rm_anova(data=long, dv='score', within='cv', subject='jd_id',
                          detailed=True, correction=True)
        r = aov.iloc[0]
        rows.append({
            'subset': subset_label, 'score_col': score_col,
            'comparison': 'all CVs',
            'test': 'RM-ANOVA', 'n': n, 'k': k,
            'statistic':   round(float(r['F']), 3),
            'p_value':     round(float(r.get('p_GG_corr', r['p_unc'])), 6),
            'effect_size': round(float(r['ng2']), 3),
            'effect_name': 'partial η²',
            'mean_diff':   '',
            'interpretation': ('large'  if r['ng2'] >= 0.14 else
                               'medium' if r['ng2'] >= 0.06 else
                               'small'  if r['ng2'] >= 0.01 else 'negligible'),
        })

    # ── Descriptive mean ─────────────────────────────────────────────────────
    for cv in cvs_present:
        vals = wide[cv].values
        lo, hi = bootstrap_ci_mean(vals)
        rows.append({
            'subset': subset_label, 'score_col': score_col,
            'comparison': cv,
            'test': 'Descriptive mean', 'n': n, 'k': 1,
            'statistic': round(float(np.mean(vals)), 3),
            'p_value': '',
            'effect_size': '',
            'effect_name': '',
            'mean_diff': '',
            'ci_low': round(lo, 3), 'ci_high': round(hi, 3),
            'median': round(float(np.median(vals)), 3),
            'interpretation': '',
        })

    return rows


def main():
    df = pd.read_csv(MASTER_CSV)
    df = df[df['error'].isna()].copy()
    print(f'Loaded {len(df)} valid rows from {MASTER_CSV.name}')

    all_rows = []
    for subset_label, source_val in SUBSET_MAP.items():
        sub = df[df['source'] == source_val]
        cvs_in = set(sub['cv'].unique())
        if not cvs_in.issuperset(set(CVS)):
            missing = set(CVS) - cvs_in
            print(f'  SKIP {subset_label}: missing {missing}')
            continue
        # FWER per subset family: 2 score_cols × 3 pairs = 6 contrasts
        for score_col in SCORE_COLS:
            all_rows += analyze(sub, subset_label, score_col, n_contrasts_in_family=6)
        print(f'  ran {subset_label}: n_jds={sub["jd_id"].nunique()}')

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_CSV, index=False)

    def section(title, mask, cols):
        print('\n' + '=' * 110)
        print(title)
        print('=' * 110)
        view = out[mask]
        existing = [c for c in cols if c in view.columns]
        print(view[existing].to_string(index=False))

    section(
        'Friedman omnibus — do CVs differ on same JDs?',
        out['test'] == 'Friedman',
        ['subset', 'score_col', 'n', 'statistic', 'p_value', 'effect_size', 'effect_name', 'interpretation'],
    )
    section(
        'Pairwise Wilcoxon (Bonferroni α=0.0167, * = significant)',
        out['test'] == 'Wilcoxon',
        ['subset', 'score_col', 'comparison', 'mean_diff', 'p_value',
         'effect_size', 'sig_after_bonferroni', 'interpretation'],
    )
    section(
        'RM-ANOVA sensitivity (fit_score_100 only, Greenhouse-Geisser corrected)',
        out['test'] == 'RM-ANOVA',
        ['subset', 'score_col', 'n', 'statistic', 'p_value', 'effect_size', 'effect_name', 'interpretation'],
    )
    section(
        'Descriptive — mean per CV (bootstrap 95% CI)',
        out['test'] == 'Descriptive mean',
        ['subset', 'score_col', 'comparison', 'statistic', 'ci_low', 'ci_high', 'median'],
    )

    print(f'\nSaved: {OUT_CSV}')


if __name__ == '__main__':
    main()
