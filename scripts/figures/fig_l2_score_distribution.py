#!/usr/bin/env python3
"""
Figure: Level 2 score distribution per CV.

2-panel row:
  - Left:  holistic_score (0–3, LLM-emitted label mapped to integer)
           — same line-plot style as Level 1 figure.
  - Right: fit_score_100 (continuous, code-computed)
           — line plot of binned proportions per CV.

Each panel shows:
  - "All" curve — pooled across the three CVs on the full JD pool
  - cv_primary, cv_hr, cv_engineer curves

Output: results/figures/L2_score_dist_curves.png
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from palette import (
    VOID, DEEP_SPACE, SLATE, MIST, GHOST,
    CV_COLORS, CV_LABELS, CV_KEYS, ALL_COLOR,
    style_axes,
)

ROOT       = Path(__file__).parent.parent.parent
MASTER_CSV = ROOT / 'results' / 'level2_master.csv'
OUT_DIR    = ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_DIR / 'L2_score_dist_curves.png'

SCORES_03  = [0, 1, 2, 3]
BINS_100   = np.arange(0, 101, 10)         # [0,10,20,...,100]
BIN_CENTERS = (BINS_100[:-1] + BINS_100[1:]) / 2


def proportion_at_scores(values):
    """Length-4 proportions for 0–3 integer scale."""
    vals = np.asarray(values, dtype=int)
    n = len(vals)
    if n == 0:
        return np.zeros(4)
    return np.array([(vals == s).sum() / n for s in SCORES_03])


def proportion_binned(values, bins):
    """Proportion in each bin for a continuous variable."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return np.zeros(len(bins) - 1)
    counts, _ = np.histogram(arr, bins=bins)
    return counts / n


def main():
    df = pd.read_csv(MASTER_CSV)

    # Use run_id == 1 to avoid triple-counting reruns; drop error rows
    df = df[(df['run_id'] == 1) & df['error'].isna()].copy()

    print(f'Loaded {len(df)} rows (run 1 only, no errors)')
    print(f'  JDs:  {df["jd_id"].nunique()}, CVs: {df["cv"].nunique()}')
    print(f'  holistic_score values: {sorted(df["holistic_score"].dropna().unique())}')

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor(VOID)

    # ─── Panel 1: holistic_score (0–3) ────────────────────────────────────────
    ax = axes[0]
    style_axes(ax)
    sub_all = df

    pooled = proportion_at_scores(sub_all['holistic_score'].values) * 100
    ax.plot(SCORES_03, pooled, color=ALL_COLOR, linewidth=2.5,
            label=f'All  (n={len(sub_all)})', zorder=4,
            marker='o', markersize=7, markeredgecolor=VOID, markeredgewidth=1.2)

    for cv in CV_KEYS:
        sub_cv = sub_all[sub_all['cv'] == cv]
        if sub_cv.empty:
            continue
        pcts = proportion_at_scores(sub_cv['holistic_score'].values) * 100
        ax.plot(SCORES_03, pcts, color=CV_COLORS[cv], linewidth=1.8, alpha=0.95,
                label=f'{CV_LABELS[cv]}  (n={len(sub_cv)})', zorder=3,
                marker='o', markersize=5, markeredgecolor=VOID, markeredgewidth=0.8)

    ax.set_xticks(SCORES_03)
    ax.set_xticklabels([str(s) for s in SCORES_03], color=MIST, fontsize=10)
    ax.set_xlabel('Score (0–3)', color=MIST, fontsize=10)
    ax.set_ylabel('% of (JD × CV) pairs', color=MIST, fontsize=11)
    ax.set_title('holistic_score  (LLM label → 0–3)', color=GHOST, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)

    # ─── Panel 2: fit_score_100 (continuous, binned) ─────────────────────────
    ax = axes[1]
    style_axes(ax)

    pooled100 = proportion_binned(sub_all['fit_score_100'].values, BINS_100) * 100
    ax.plot(BIN_CENTERS, pooled100, color=ALL_COLOR, linewidth=2.5,
            label=f'All  (n={len(sub_all)})', zorder=4,
            marker='o', markersize=6, markeredgecolor=VOID, markeredgewidth=1.0)

    for cv in CV_KEYS:
        sub_cv = sub_all[sub_all['cv'] == cv]
        if sub_cv.empty:
            continue
        pcts = proportion_binned(sub_cv['fit_score_100'].values, BINS_100) * 100
        ax.plot(BIN_CENTERS, pcts, color=CV_COLORS[cv], linewidth=1.8, alpha=0.95,
                label=f'{CV_LABELS[cv]}  (n={len(sub_cv)})', zorder=3,
                marker='o', markersize=4, markeredgecolor=VOID, markeredgewidth=0.6)

    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xticklabels([str(x) for x in np.arange(0, 101, 20)], color=MIST, fontsize=10)
    ax.set_xlabel('fit_score_100  (bin width = 10)', color=MIST, fontsize=10)
    ax.set_title('fit_score_100  (formula-derived 0–100)', color=GHOST, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(50, ax.get_ylim()[1]))

    # ─── Single legend ───────────────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=4,
               facecolor=DEEP_SPACE, edgecolor=SLATE,
               labelcolor=GHOST, fontsize=10,
               frameon=True)

    fig.suptitle('Level 2 — score distribution by CV (single prompt: L2 P0)',
                 color=GHOST, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=VOID)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
