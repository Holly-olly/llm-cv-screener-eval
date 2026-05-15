#!/usr/bin/env python3
"""
Level 2 — criterion validity vs human holistic labels.

Reads results/level2_master.csv. Filters cv == cv_primary, source == main
(where human labels exist). Aggregates 3 runs per JD: median LLM score.

Compares LLM holistic_score (0–3) to human_holistic_label (0–3).

Metrics (identical to Level 1):
  - Spearman ρ + 95% CI   (primary: ordinal × ordinal)
  - Pearson r + 95% CI    (secondary)
  - Weighted Cohen κ + 95% CI (linear weights, integer 0–3)
  - Bias (LLM − human) + Wilcoxon p + Cohen's d_paired
  - MAE + 95% CI
  - % exact match  + Wilson 95% CI
  - % within ±1    + Wilson 95% CI

Output: results/analysis/level2_validity.csv
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pingouin as pg
from pathlib import Path
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

ROOT       = Path(__file__).parent.parent
MASTER_CSV = ROOT / 'results' / 'level2_master.csv'
OUT_DIR    = ROOT / 'results' / 'analysis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV    = OUT_DIR / 'level2_validity.csv'

LABELS   = [0, 1, 2, 3]
N_BOOT   = 2000
RNG_SEED = 42


def aggregate_runs(df, score_col):
    agg = df.groupby('jd_id').agg(
        llm_score = (score_col, 'median'),
        human     = ('human_holistic_label', 'first'),
    ).reset_index()
    agg = agg[agg['llm_score'].notna() & agg['human'].notna()].copy()
    agg['human']         = agg['human'].astype(int)
    agg['llm_score_int'] = agg['llm_score'].round().astype(int).clip(0, 3)
    return agg


def compute_corr(x, y, method):
    r = pg.corr(x, y, method=method)
    ci = r['CI95'].iloc[0]
    return {
        'value':   round(float(r['r'].iloc[0]), 3),
        'ci_low':  round(float(ci[0]), 3),
        'ci_high': round(float(ci[1]), 3),
        'p_value': round(float(r['p_val'].iloc[0]), 4),
    }


def kappa_with_ci(y_llm, y_hum, weights='linear', n_boot=N_BOOT):
    kappa = cohen_kappa_score(y_llm, y_hum, weights=weights, labels=LABELS)
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_llm)
    y_llm_arr, y_hum_arr = np.asarray(y_llm), np.asarray(y_hum)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            k = cohen_kappa_score(y_llm_arr[idx], y_hum_arr[idx],
                                  weights=weights, labels=LABELS)
            if not np.isnan(k):
                boots.append(k)
        except Exception:
            pass
    if not boots:
        return {'value': round(kappa, 3), 'ci_low': np.nan, 'ci_high': np.nan}
    boots = np.array(boots)
    return {
        'value':   round(float(kappa), 3),
        'ci_low':  round(float(np.percentile(boots, 2.5)), 3),
        'ci_high': round(float(np.percentile(boots, 97.5)), 3),
    }


def bias_with_effect(llm, hum):
    diff = np.asarray(llm) - np.asarray(hum)
    n = len(diff)
    bias = diff.mean()
    sd_diff = diff.std(ddof=1)
    d = bias / sd_diff if sd_diff > 0 else np.nan
    rng = np.random.default_rng(RNG_SEED)
    boots = np.array([np.random.default_rng(s).choice(diff, n, replace=True).mean()
                      for s in rng.integers(0, 1_000_000, N_BOOT)])
    try:
        _, p = stats.wilcoxon(diff)
    except Exception:
        p = np.nan
    return {
        'value':   round(float(bias), 3),
        'ci_low':  round(float(np.percentile(boots, 2.5)), 3),
        'ci_high': round(float(np.percentile(boots, 97.5)), 3),
        'p_value': round(float(p), 4) if not np.isnan(p) else '',
        'cohens_d': round(float(d), 3) if not np.isnan(d) else '',
    }


def bootstrap_ci(values, statistic_fn=np.mean, n_boot=N_BOOT):
    arr = np.asarray(values)
    rng = np.random.default_rng(RNG_SEED)
    n = len(arr)
    boots = np.array([statistic_fn(arr[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def interpret_corr(v):
    if pd.isna(v): return ''
    a = abs(v)
    if a < 0.3: return 'weak'
    if a < 0.5: return 'moderate'
    if a < 0.7: return 'strong'
    return 'very strong'


def interpret_kappa(v):
    if pd.isna(v): return ''
    if v < 0.20: return 'slight'
    if v < 0.40: return 'fair'
    if v < 0.60: return 'moderate'
    if v < 0.80: return 'substantial'
    return 'almost perfect'


def interpret_d(v):
    if pd.isna(v) or v == '': return ''
    a = abs(v)
    if a < 0.2: return 'negligible'
    if a < 0.5: return 'small'
    if a < 0.8: return 'medium'
    return 'large'


def analyze_one(df, prompt_label, score_col='holistic_score'):
    """Run all metrics on one filtered slice. Returns list of row dicts."""
    rows = []
    agg = aggregate_runs(df, score_col)
    n = len(agg)
    if n < 3:
        return rows

    sp = compute_corr(agg['llm_score'], agg['human'], 'spearman')
    pe = compute_corr(agg['llm_score'], agg['human'], 'pearson')
    kw = kappa_with_ci(agg['llm_score_int'], agg['human'], weights='linear')

    exact_arr = (agg['llm_score_int'] == agg['human']).astype(int)
    exact = exact_arr.mean() * 100
    ex_lo, ex_hi = proportion_confint(int(exact_arr.sum()), n, method='wilson')

    within1_arr = ((agg['llm_score'] - agg['human']).abs() <= 1).astype(int)
    within1 = within1_arr.mean() * 100
    w1_lo, w1_hi = proportion_confint(int(within1_arr.sum()), n, method='wilson')

    abs_err = (agg['llm_score'] - agg['human']).abs()
    mae = abs_err.mean()
    mae_lo, mae_hi = bootstrap_ci(abs_err.values, np.mean)

    bias = bias_with_effect(agg['llm_score'], agg['human'])

    for name, r in [('Spearman ρ', sp), ('Pearson r', pe), ('Weighted κ', kw)]:
        interp = interpret_kappa(r['value']) if 'κ' in name else interpret_corr(r['value'])
        rows.append({'scale': 'holistic_score_0-3', 'prompt': prompt_label,
                     'metric': name, 'n': n,
                     'value': r['value'], 'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
                     'p_value': r.get('p_value', ''), 'effect_size': '',
                     'interpretation': interp})
    rows.append({'scale': 'holistic_score_0-3', 'prompt': prompt_label,
                 'metric': 'Bias (LLM−human)', 'n': n,
                 'value': bias['value'], 'ci_low': bias['ci_low'], 'ci_high': bias['ci_high'],
                 'p_value': bias['p_value'], 'effect_size': bias['cohens_d'],
                 'interpretation': f"Cohen's d: {interpret_d(bias['cohens_d'])}"})
    rows.append({'scale': 'holistic_score_0-3', 'prompt': prompt_label,
                 'metric': 'MAE', 'n': n,
                 'value': round(mae, 3), 'ci_low': round(mae_lo, 3), 'ci_high': round(mae_hi, 3),
                 'p_value': '', 'effect_size': '',
                 'interpretation': 'ordinal — interpret with caution'})
    rows.append({'scale': 'holistic_score_0-3', 'prompt': prompt_label,
                 'metric': '% exact match', 'n': n,
                 'value': round(exact, 1), 'ci_low': round(ex_lo * 100, 1), 'ci_high': round(ex_hi * 100, 1),
                 'p_value': '', 'effect_size': '', 'interpretation': ''})
    rows.append({'scale': 'holistic_score_0-3', 'prompt': prompt_label,
                 'metric': '% within ±1', 'n': n,
                 'value': round(within1, 1), 'ci_low': round(w1_lo * 100, 1), 'ci_high': round(w1_hi * 100, 1),
                 'p_value': '', 'effect_size': '', 'interpretation': ''})
    return rows


def main():
    df = pd.read_csv(MASTER_CSV)
    df = df[(df['cv'] == 'cv_primary') & df['human_holistic_label'].notna() & df['error'].isna()].copy()
    print(f'Filtered: {len(df)} rows for cv_primary with human labels '
          f'({df["jd_id"].nunique()} unique JDs × {df["run_id"].nunique()} runs)')

    rows = analyze_one(df, prompt_label='L2_P0', score_col='holistic_score')

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print('\n' + '=' * 100)
    print('holistic_score (0–3) — LLM vs human')
    print('=' * 100)
    print(out.to_string(index=False))

    print(f'\nSaved: {OUT_CSV}')


if __name__ == '__main__':
    main()
