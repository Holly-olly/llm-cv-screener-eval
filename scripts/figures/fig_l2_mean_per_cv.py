#!/usr/bin/env python3
"""
Figure: Level 2 — mean score per CV across three JD pools.

Two side-by-side panels:
  - Left:  mean `holistic_score` (0–3, LLM label → integer)
  - Right: mean `fit_score_100` (continuous, code-computed)

Each panel: three groups (one per JD pool) × three bars (one per CV) with
bootstrap 95 % CIs. The `main` pool is filtered to JDs where `cv_primary`
in Level 1 was labeled human_holistic_label ≥ 2 (n = 16), to match the
discriminant-validity sample used in the unified analysis.

Output: results/figures/L2_mean_per_cv.png
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from palette import (
    VOID, DEEP_SPACE, SLATE, MIST, GHOST,
    CV_COLORS, CV_LABELS_SHORT, CV_KEYS,
    POOL_KEYS, POOL_LABELS,
    style_axes,
)

ROOT          = Path(__file__).parent.parent.parent
MASTER_CSV_L2 = ROOT / 'results' / 'level2_master.csv'
MASTER_CSV_L1 = ROOT / 'results' / 'level1_master.csv'  # for human labels
OUT_DIR       = ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH      = OUT_DIR / 'L2_mean_per_cv.png'

N_BOOT = 2000
RNG_SEED = 42


def bootstrap_ci_mean(values, n_boot=N_BOOT, seed=RNG_SEED):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_per_jd_cv(df, score_col):
    """Median over 3 runs → one value per (jd_id, cv)."""
    return df.groupby(['jd_id', 'cv'])[score_col].median().reset_index()


def panel_for_score(ax, df_l2, jds_relevant, score_col, ylabel, ylim):
    """Draw one panel (grouped bars: 3 pools × 3 CVs) into `ax`."""
    style_axes(ax)

    pool_means, pool_lows, pool_highs, pool_ns = {}, {}, {}, {}

    for pool in POOL_KEYS:
        if pool == 'main':
            sub = df_l2[df_l2['jd_id'].isin(jds_relevant)]
        else:
            sub = df_l2[df_l2['source'] == pool]
        agg = aggregate_per_jd_cv(sub, score_col)
        pool_means[pool], pool_lows[pool], pool_highs[pool] = {}, {}, {}
        pool_ns[pool] = agg['jd_id'].nunique()
        for cv in CV_KEYS:
            vals = agg.loc[agg['cv'] == cv, score_col].dropna().values
            pool_means[pool][cv] = float(np.mean(vals)) if len(vals) else np.nan
            lo, hi = bootstrap_ci_mean(vals)
            pool_lows[pool][cv]  = lo
            pool_highs[pool][cv] = hi

    n_pools = len(POOL_KEYS)
    n_cvs   = len(CV_KEYS)
    bar_w   = 0.25
    group_centers = np.arange(n_pools)
    offsets = (np.arange(n_cvs) - (n_cvs - 1) / 2) * bar_w

    for j, cv in enumerate(CV_KEYS):
        xs    = group_centers + offsets[j]
        means = np.array([pool_means[p][cv] for p in POOL_KEYS])
        lows  = np.array([pool_lows[p][cv]  for p in POOL_KEYS])
        highs = np.array([pool_highs[p][cv] for p in POOL_KEYS])
        err_lo = np.maximum(0, means - lows)
        err_hi = np.maximum(0, highs - means)

        ax.bar(xs, means, width=bar_w, color=CV_COLORS[cv], alpha=0.88,
               yerr=[err_lo, err_hi], capsize=3,
               error_kw={'ecolor': MIST, 'elinewidth': 1.0},
               label=CV_LABELS_SHORT[cv], zorder=3,
               edgecolor=VOID, linewidth=0.8)
        for xi, m in zip(xs, means):
            if not np.isnan(m):
                ax.text(xi, m + (ylim[1] - ylim[0]) * 0.02, f'{m:.2f}',
                        ha='center', va='bottom', color=GHOST,
                        fontsize=8, fontweight='bold')

    pool_tick_labels = [
        f'{POOL_LABELS[p].split(" (")[0]}\n(n={pool_ns[p]})'
        for p in POOL_KEYS
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(pool_tick_labels, color=MIST, fontsize=10)
    ax.set_ylabel(ylabel, color=MIST, fontsize=11)
    ax.set_ylim(*ylim)


def main():
    l2 = pd.read_csv(MASTER_CSV_L2)
    l2 = l2[l2['error'].isna()].copy()
    print(f'Level 2 rows (no errors): {len(l2)}  ({l2["jd_id"].nunique()} unique JDs)')

    # Filtered main subset — using L1 labels (same rule as discriminant analysis)
    l1 = pd.read_csv(MASTER_CSV_L1)
    jds_relevant = l1[
        (l1['source'] == 'main') &
        (l1['cv'] == 'cv_primary') &
        (l1['human_holistic_label'].fillna(0) >= 2)
    ]['jd_id'].unique()
    print(f'Filtered main: n = {len(jds_relevant)} JDs')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.patch.set_facecolor(VOID)

    panel_for_score(axes[0], l2, jds_relevant,
                    score_col='holistic_score',
                    ylabel='Mean holistic_score (0–3) ± 95 % CI',
                    ylim=(0, 3.4))
    axes[0].set_title('holistic_score  (LLM label → 0–3)',
                      color=GHOST, fontsize=12, fontweight='bold')

    panel_for_score(axes[1], l2, jds_relevant,
                    score_col='fit_score_100',
                    ylabel='Mean fit_score_100 ± 95 % CI',
                    ylim=(0, 100))
    axes[1].set_title('fit_score_100  (formula-derived 0–100)',
                      color=GHOST, fontsize=12, fontweight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=3,
               facecolor=DEEP_SPACE, edgecolor=SLATE,
               labelcolor=GHOST, fontsize=10, frameon=True)

    fig.suptitle('Level 2 (P0) — mean score per CV across JD pools  ·  flip test visual',
                 color=GHOST, fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=VOID)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
