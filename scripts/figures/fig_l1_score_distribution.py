#!/usr/bin/env python3
"""
Figure: Level 1 score distribution (0–3 scale) per prompt and per CV.

3-panel row (P0, P1, P2). Each panel:
  - "All" curve — pooled across the three CVs on the full JD pool
  - cv_primary, cv_hr, cv_engineer curves

Curves are line plots of the proportion at each integer score (0, 1, 2, 3),
not bars, to make cross-CV shape comparison easy.

Output: results/figures/L1_score_dist_curves.png
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
MASTER_CSV = ROOT / 'results' / 'level1_master.csv'
OUT_DIR    = ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = OUT_DIR / 'L1_score_dist_curves.png'

SCORES  = [0, 1, 2, 3]
PROMPTS = ['P0', 'P1', 'P2']


def proportion_at_scores(values):
    """Return a length-4 array of proportions at scores 0, 1, 2, 3."""
    vals = np.asarray(values, dtype=int)
    n = len(vals)
    if n == 0:
        return np.zeros(4)
    return np.array([(vals == s).sum() / n for s in SCORES])


def main():
    df = pd.read_csv(MASTER_CSV)

    # Use run_id == 1 to avoid triple-counting the 3 reruns
    df = df[df['run_id'] == 1].copy()

    print(f'Loaded {len(df)} rows (run 1 only)')
    for p in PROMPTS:
        sub = df[df['prompt'] == p]
        print(f'  {p}: n={len(sub)} ({sub["jd_id"].nunique()} JDs × {sub["cv"].nunique()} CVs)')

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    fig.patch.set_facecolor(VOID)

    for ax, prompt in zip(axes, PROMPTS):
        style_axes(ax)
        sub_all = df[df['prompt'] == prompt]
        if sub_all.empty:
            ax.text(0.5, 0.5, 'no data', color=MIST, transform=ax.transAxes, ha='center')
            ax.set_title(f'{prompt}', color=GHOST, fontsize=11)
            continue

        # --- Pooled "All" curve ------------------------------------------------
        pooled = proportion_at_scores(sub_all['score'].values)
        ax.plot(SCORES, pooled * 100, color=ALL_COLOR, linewidth=2.5,
                label=f'All  (n={len(sub_all)})', zorder=4,
                marker='o', markersize=7, markeredgecolor=VOID, markeredgewidth=1.2)

        # --- Per-CV curves -----------------------------------------------------
        for cv in CV_KEYS:
            sub_cv = sub_all[sub_all['cv'] == cv]
            if sub_cv.empty:
                continue
            pcts = proportion_at_scores(sub_cv['score'].values) * 100
            color = CV_COLORS[cv]
            ax.plot(SCORES, pcts, color=color, linewidth=1.8, alpha=0.95,
                    label=f'{CV_LABELS[cv]}  (n={len(sub_cv)})', zorder=3,
                    marker='o', markersize=5, markeredgecolor=VOID, markeredgewidth=0.8)

        ax.set_xticks(SCORES)
        ax.set_xticklabels([str(s) for s in SCORES], color=MIST, fontsize=10)
        ax.set_xlabel('Score (0–3)', color=MIST, fontsize=10)
        n_jds = sub_all['jd_id'].nunique()
        ax.set_title(f'{prompt}   ({n_jds} unique JDs)', color=GHOST, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)

        if prompt == 'P0':
            ax.set_ylabel('% of (JD × CV) pairs', color=MIST, fontsize=11)

    # --- Single legend for the whole figure ------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=4,
               facecolor=DEEP_SPACE, edgecolor=SLATE,
               labelcolor=GHOST, fontsize=10,
               frameon=True)

    fig.suptitle('Level 1 — score distribution per prompt, by CV',
                 color=GHOST, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=VOID)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
