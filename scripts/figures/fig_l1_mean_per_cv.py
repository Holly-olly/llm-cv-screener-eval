#!/usr/bin/env python3
"""
Figure: Level 1 P2 — mean score per CV across three JD pools.

Two side-by-side panels:
  - Left:  mean `score` (0–3)
  - Right: mean `score_100` (0–100)

Each panel shows three groups (one per JD pool) and three bars per group
(one per CV), with bootstrap 95 % CIs. The `main` pool is filtered to JDs
where `cv_primary` was labeled human_holistic_label ≥ 2 (n = 16), to match
the discriminant-validity sample.

Output: results/figures/L1P2_mean_per_cv.png
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

ROOT       = Path(__file__).parent.parent.parent
MASTER_CSV = ROOT / 'results' / 'level1_master.csv'
OUT_DIR    = ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_DIR / 'L1P2_mean_per_cv.png'

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


def panel_for_score(ax, df_p2, jds_relevant, score_col, ylabel, ylim):
    """Draw one panel (grouped bars: 3 pools × 3 CVs) into `ax`."""
    style_axes(ax)

    pool_means = {}
    pool_lows  = {}
    pool_highs = {}
    pool_ns    = {}

    for pool in POOL_KEYS:
        if pool == 'main':
            sub = df_p2[df_p2['jd_id'].isin(jds_relevant)]
        else:
            sub = df_p2[df_p2['source'] == pool]
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
        means = [pool_means[p][cv] for p in POOL_KEYS]
        lows  = [pool_lows[p][cv]  for p in POOL_KEYS]
        highs = [pool_highs[p][cv] for p in POOL_KEYS]
        means = np.array(means); lows = np.array(lows); highs = np.array(highs)
        err_lo = np.maximum(0, means - lows)
        err_hi = np.maximum(0, highs - means)

        ax.bar(xs, means, width=bar_w, color=CV_COLORS[cv], alpha=0.88,
               yerr=[err_lo, err_hi], capsize=3,
               error_kw={'ecolor': MIST, 'elinewidth': 1.0},
               label=CV_LABELS_SHORT[cv], zorder=3, edgecolor=VOID, linewidth=0.8)
        # value labels
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
    df = pd.read_csv(MASTER_CSV)
    df_p2 = df[df['prompt'] == 'P2'].copy()
    print(f'Level 1 P2 rows: {len(df_p2)}  ({df_p2["jd_id"].nunique()} unique JDs)')

    # filtered main subset (cv_primary human label ≥ 2)
    jds_relevant = df[
        (df['source'] == 'main') &
        (df['cv'] == 'cv_primary') &
        (df['human_holistic_label'].fillna(0) >= 2)
    ]['jd_id'].unique()
    print(f'Filtered main: n = {len(jds_relevant)} JDs')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.patch.set_facecolor(VOID)

    panel_for_score(axes[0], df_p2, jds_relevant,
                    score_col='score',
                    ylabel='Mean score (0–3) ± 95 % CI',
                    ylim=(0, 3.4))
    axes[0].set_title('Score 0–3', color=GHOST, fontsize=12, fontweight='bold')

    panel_for_score(axes[1], df_p2, jds_relevant,
                    score_col='score_100',
                    ylabel='Mean score (0–100) ± 95 % CI',
                    ylim=(0, 100))
    axes[1].set_title('Score 0–100', color=GHOST, fontsize=12, fontweight='bold')

    # single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=3,
               facecolor=DEEP_SPACE, edgecolor=SLATE,
               labelcolor=GHOST, fontsize=10, frameon=True)

    fig.suptitle('Level 1 (P2) — mean score per CV across JD pools  ·  flip test visual',
                 color=GHOST, fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=VOID)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
