#!/usr/bin/env python3
"""
Level 3 — Step 6: Visualise construct validity of LLM tags.

Two figures, both in the Neural Haze palette used elsewhere in the project:

  fig_scatter_skills_vs_experience.png
    Scatter of every non-`other` segment in (sim_skills, sim_experience)
    space, coloured by assigned tag (skills / experience / mixed). Group
    centroids overlaid as larger markers; diagonal y=x as reference.

  fig_similarity_distributions.png
    Three panels — one per prototype (sim_skills, sim_experience, sim_edu).
    In each panel, a KDE curve per assigned tag shows where its segments
    land on that prototype's similarity axis.

Reads:
  - results/level3/segment_similarities.csv

Writes:
  - results/level3/figures/fig_scatter_skills_vs_experience.png
  - results/level3/figures/fig_similarity_distributions.png

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/06_visualize_validity.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from palette import (
    VOID, GHOST, MIST, SLATE,
    CYAN, VIOLET, MAGENTA, ACID_LIME,
    style_axes,
)

SIM_CSV = ROOT / "results" / "level3" / "segment_similarities.csv"
FIG_DIR = ROOT / "results" / "level3" / "figures"

TAG_COLORS = {
    "skills":     CYAN,
    "experience": VIOLET,
    "mixed":      MAGENTA,
    "education":  ACID_LIME,
}
TAG_ORDER = ["skills", "experience", "mixed", "education"]


def _set_global_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": VOID,
        "axes.facecolor":   VOID,
        "axes.edgecolor":   SLATE,
        "axes.labelcolor":  GHOST,
        "text.color":       GHOST,
        "xtick.color":      MIST,
        "ytick.color":      MIST,
        "axes.titlecolor":  GHOST,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "axes.labelsize":   10,
        "legend.facecolor": VOID,
        "legend.edgecolor": SLATE,
        "legend.labelcolor": GHOST,
        "font.family":      "DejaVu Sans",
    })


# ── Figure 1 — Scatter ──────────────────────────────────────────────────────
def fig_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    sub = df[df["tag"].isin(["skills", "experience", "mixed"])]
    for tag in ["skills", "experience", "mixed"]:
        rows = sub[sub["tag"] == tag]
        ax.scatter(
            rows["sim_skills"], rows["sim_experience"],
            s=22, alpha=0.55, color=TAG_COLORS[tag],
            edgecolors="none",
            label=f"{tag}  (n={len(rows)})",
            zorder=2,
        )

    # Centroids
    for tag in ["skills", "experience", "mixed"]:
        rows = sub[sub["tag"] == tag]
        if not len(rows):
            continue
        ax.scatter(
            rows["sim_skills"].mean(), rows["sim_experience"].mean(),
            s=260, color=TAG_COLORS[tag],
            edgecolors=GHOST, linewidths=1.8,
            marker="o", zorder=4,
        )

    # Diagonal reference y = x
    lo = min(sub["sim_skills"].min(), sub["sim_experience"].min())
    hi = max(sub["sim_skills"].max(), sub["sim_experience"].max())
    pad = 0.02 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=MIST, linestyle="--", linewidth=1.0, alpha=0.5, zorder=1,
            label="y = x  (equal similarity)")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Cosine similarity to Skills prototype")
    ax.set_ylabel("Cosine similarity to Experience prototype")
    ax.set_title("Segment positions in (Skills, Experience) similarity space\n"
                 "large circles = group centroids")
    style_axes(ax)

    leg = ax.legend(loc="lower right", frameon=True, fontsize=9)
    leg.get_frame().set_alpha(0.9)

    fig.tight_layout()
    out = FIG_DIR / "fig_scatter_skills_vs_experience.png"
    fig.savefig(out, dpi=160, facecolor=VOID)
    plt.close(fig)
    print(f"✓ {out.relative_to(ROOT)}")


# ── Figure 2 — Distributions ────────────────────────────────────────────────
def fig_distributions(df: pd.DataFrame) -> None:
    proto_cols = ["sim_skills", "sim_experience", "sim_education"]
    proto_titles = {
        "sim_skills":     "Similarity to Skills prototype",
        "sim_experience": "Similarity to Experience prototype",
        "sim_education":  "Similarity to Education prototype",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    # Use a common x-grid for KDE evaluation so panels are comparable.
    all_sims = pd.concat([df[c] for c in proto_cols])
    x_min, x_max = all_sims.min(), all_sims.max()
    pad = 0.03 * (x_max - x_min)
    grid = np.linspace(x_min - pad, x_max + pad, 400)

    for ax, col in zip(axes, proto_cols):
        for tag in TAG_ORDER:
            rows = df[df["tag"] == tag]
            if len(rows) < 5:
                continue
            kde = gaussian_kde(rows[col].to_numpy())
            ax.plot(
                grid, kde(grid),
                color=TAG_COLORS[tag], linewidth=2.0, alpha=0.9,
                label=f"{tag} (n={len(rows)})", zorder=3,
            )
            ax.fill_between(grid, 0, kde(grid),
                            color=TAG_COLORS[tag], alpha=0.10, zorder=2)
            # Mean marker as a short vertical bar at top
            mean_x = rows[col].mean()
            ax.axvline(mean_x, color=TAG_COLORS[tag], linestyle=":",
                       linewidth=1.0, alpha=0.7, zorder=2)

        ax.set_title(proto_titles[col])
        ax.set_xlabel("cosine similarity")
        style_axes(ax)

    axes[0].set_ylabel("density")
    axes[-1].legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle("Distribution of cosine similarity per prototype, by assigned tag",
                 color=GHOST, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_similarity_distributions.png"
    fig.savefig(out, dpi=160, facecolor=VOID, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.relative_to(ROOT)}")


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SIM_CSV)
    print(f"Loaded {len(df)} segments from {SIM_CSV.relative_to(ROOT)}")
    print(f"Tag counts: {df['tag'].value_counts().to_dict()}\n")

    _set_global_style()
    fig_scatter(df)
    fig_distributions(df)


if __name__ == "__main__":
    main()
