#!/usr/bin/env python3
"""
Embed job titles from role_titles.csv, compute cosine similarity to 3 CV profiles,
save sim_cv1/sim_cv2/sim_cv3 columns to CSV, and produce two charts:
  - figures/role_title_similarity_pairs.png  — 3 pairwise scatter plots
  - figures/role_title_similarity_ternary.png — ternary triangle plot

CV anchors (embedded inline, not in CSV):
  CV-1: Assessment Scientist  Psychometrics AI Evaluation
  CV-2: Solution Technical Architect
  CV-3: Head of People & Organizational Coach

Usage:
    python3 embed_role_titles.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import ternary
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT         = Path(__file__).parent.parent
CSV_PATH     = ROOT / 'data' / 'role_titles.csv'
FIG_PAIRS    = ROOT / 'figures' / 'role_title_similarity_pairs.png'
FIG_TERNARY  = ROOT / 'figures' / 'role_title_similarity_ternary.png'
FIG_PAIRS.parent.mkdir(exist_ok=True)

MODEL = 'all-MiniLM-L6-v2'

CV1 = 'Assessment Scientist  Psychometrics AI Evaluation'
CV2 = 'Solution Technical Architect'
CV3 = 'Head of People & Organizational Coach'

CV_LABELS = ['Assessment Scientist', 'Solution Architect', 'HR & Org Coach']

# Neural Haze palette
VOID        = '#0B0A14'
DEEP_SPACE  = '#1C1928'
SLATE       = '#6E6A88'
MIST        = '#C5C3D6'
GHOST_WHITE = '#F0EFF8'
ACID_LIME   = '#C8F135'
VIOLET      = '#9B30FF'
CYAN        = '#00F5D4'
MAGENTA     = '#FF2D78'

COLORS = {
    'labeled':   ACID_LIME,
    'unlabeled': SLATE,
    'cv':        MAGENTA,
}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    with open(CSV_PATH, encoding='utf-8') as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))

    titles  = [r['job_title'].strip() for r in rows]
    sources = [r['source'].strip()    for r in rows]
    print(f'Titles: {len(titles)}  (labeled: {sources.count("labeled")}  unlabeled: {sources.count("unlabeled")})')
    print(f'Model: {MODEL}\n')

    model = SentenceTransformer(MODEL, local_files_only=True)

    all_texts = titles + [CV1, CV2, CV3]
    all_emb   = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)

    emb     = all_emb[:len(titles)]
    cv1_vec = all_emb[-3]
    cv2_vec = all_emb[-2]
    cv3_vec = all_emb[-1]

    sim_cv1 = [round(cosine_sim(e, cv1_vec), 4) for e in emb]
    sim_cv2 = [round(cosine_sim(e, cv2_vec), 4) for e in emb]
    sim_cv3 = [round(cosine_sim(e, cv3_vec), 4) for e in emb]

    # ── save to CSV ────────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        row['sim_cv1'] = sim_cv1[i]
        row['sim_cv2'] = sim_cv2[i]
        row['sim_cv3'] = sim_cv3[i]

    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['job_title', 'source', 'sim_cv1', 'sim_cv2', 'sim_cv3'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved to: {CSV_PATH}')

    # pairwise scatter subplots ─────────────────────────────────
    pairs = [
        (sim_cv1, sim_cv2, CV_LABELS[0], CV_LABELS[1]),
        (sim_cv1, sim_cv3, CV_LABELS[0], CV_LABELS[2]),
        (sim_cv2, sim_cv3, CV_LABELS[1], CV_LABELS[2]),
    ]
    cv_sims = [
        cosine_sim(cv1_vec, cv2_vec),
        cosine_sim(cv1_vec, cv3_vec),
        cosine_sim(cv2_vec, cv3_vec),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(VOID)

    for ax, (xs_all, ys_all, xlabel, ylabel), anchor_sim in zip(axes, pairs, cv_sims):
        ax.set_facecolor(DEEP_SPACE)
        for src in ('unlabeled', 'labeled'):
            xs = [xs_all[i] for i, s in enumerate(sources) if s == src]
            ys = [ys_all[i] for i, s in enumerate(sources) if s == src]
            ax.scatter(xs, ys, color=COLORS[src],
                       alpha=0.5 if src == 'unlabeled' else 0.9,
                       s=30 if src == 'unlabeled' else 55,
                       zorder=2 if src == 'labeled' else 1,
                       label=src.capitalize())
        ax.scatter([1.0], [anchor_sim], color=MAGENTA, s=180, marker='*', zorder=5)
        ax.scatter([anchor_sim], [1.0], color=MAGENTA, s=180, marker='*', zorder=5, label='CV')
        ax.set_xlabel(f'↔ {xlabel}', color=MIST, fontsize=9)
        ax.set_ylabel(f'↔ {ylabel}', color=MIST, fontsize=9)
        ax.set_title(f'{xlabel}\nvs {ylabel}', color=GHOST_WHITE, fontsize=9)
        ax.tick_params(colors=SLATE)
        for spine in ax.spines.values():
            spine.set_edgecolor(SLATE)
        ax.grid(color=SLATE, linestyle='--', linewidth=0.4, alpha=0.3)
        ax.legend(facecolor=DEEP_SPACE, edgecolor=SLATE, labelcolor=GHOST_WHITE, fontsize=8)

    fig.suptitle('Job Title Similarity — Pairwise CV Comparisons', color=GHOST_WHITE, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_PAIRS, dpi=150, bbox_inches='tight')
    print(f'Pairs chart saved: {FIG_PAIRS}')
    plt.show()

if __name__ == '__main__':
    main()
