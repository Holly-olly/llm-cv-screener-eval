#!/usr/bin/env python3
"""
Level 3 — Step 7: Segment-length distribution charts.

Two figures in the Neural Haze palette:

  fig_seglen_per_tag.png
    Per-segment length distribution (chars), one KDE curve per tag.
    Uses ALL segments across the 72 labelled JDs (every tag, including
    `other`).

  fig_seglen_per_jd_means.png
    Per-JD MEAN segment length, one KDE curve per tag. For each JD and tag
    we compute the mean character length of the segments with that tag,
    then plot the distribution of those per-JD means.

Reads:
  - results/level3/llm_labelled_json/*.csv

Writes:
  - results/level3/figures/fig_seglen_per_tag.png
  - results/level3/figures/fig_seglen_per_jd_means.png

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/07_length_charts.py
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

LBL_DIR = ROOT / "results" / "level3" / "llm_labelled_json"
FIG_DIR = ROOT / "results" / "level3" / "figures"

TAG_COLORS = {
    "skills":     CYAN,
    "experience": VIOLET,
    "mixed":      MAGENTA,
    "education":  ACID_LIME,
    "other":      MIST,
}
TAG_ORDER = ["skills", "experience", "mixed", "education", "other"]


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


def load_all_segments() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(LBL_DIR.glob("*.csv")):
        jd_id = csv_path.stem
        df = pd.read_csv(csv_path, keep_default_na=False)
        df["jd_id"] = jd_id
        df["text_length"] = df["text"].astype(str).str.len()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _kde_panel(ax, data_by_tag: dict[str, np.ndarray], xlim: tuple[float, float],
               title: str, xlabel: str) -> None:
    grid = np.linspace(xlim[0], xlim[1], 400)
    for tag in TAG_ORDER:
        arr = data_by_tag.get(tag)
        if arr is None or len(arr) < 5:
            continue
        try:
            kde = gaussian_kde(arr)
        except (np.linalg.LinAlgError, ValueError):
            continue
        ax.plot(grid, kde(grid),
                color=TAG_COLORS[tag], linewidth=2.0, alpha=0.9,
                label=f"{tag} (n={len(arr)})", zorder=3)
        ax.fill_between(grid, 0, kde(grid),
                        color=TAG_COLORS[tag], alpha=0.10, zorder=2)
        ax.axvline(np.mean(arr), color=TAG_COLORS[tag],
                   linestyle=":", linewidth=1.0, alpha=0.7, zorder=2)

    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    style_axes(ax)
    leg = ax.legend(loc="upper right", fontsize=9, frameon=True)
    leg.get_frame().set_alpha(0.9)


def fig_seglen_per_tag(df: pd.DataFrame) -> None:
    data = {t: df.loc[df["tag"] == t, "text_length"].to_numpy() for t in TAG_ORDER}
    # Clip x to the 95th percentile of pooled data so a few long sentences
    # don't squash the curves into the leftmost column.
    upper = float(np.quantile(df["text_length"], 0.95))
    upper = max(upper, 200)   # ensure decent range

    fig, ax = plt.subplots(figsize=(11, 5.2))
    _kde_panel(ax, data, xlim=(0, upper),
               title=f"Segment length distribution per tag  "
                     f"(x-axis clipped at the 95th percentile = {upper:.0f} chars)",
               xlabel="segment length (characters)")
    fig.tight_layout()
    out = FIG_DIR / "fig_seglen_per_tag.png"
    fig.savefig(out, dpi=160, facecolor=VOID, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.relative_to(ROOT)}")


def fig_seglen_per_jd_means(df: pd.DataFrame) -> None:
    per_jd_tag = (
        df.groupby(["jd_id", "tag"])["text_length"]
          .mean()
          .reset_index()
    )
    data = {
        t: per_jd_tag.loc[per_jd_tag["tag"] == t, "text_length"].to_numpy()
        for t in TAG_ORDER
    }
    pooled = per_jd_tag["text_length"]
    upper = float(np.quantile(pooled, 0.95))
    upper = max(upper, 200)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    _kde_panel(ax, data, xlim=(0, upper),
               title=f"Distribution of per-JD MEAN segment length, by tag  "
                     f"(each density point is one JD; x clipped at P95 = {upper:.0f})",
               xlabel="mean segment length within a JD (characters)")
    fig.tight_layout()
    out = FIG_DIR / "fig_seglen_per_jd_means.png"
    fig.savefig(out, dpi=160, facecolor=VOID, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.relative_to(ROOT)}")


def print_summary(df: pd.DataFrame) -> None:
    print("\nPer-tag length stats across all segments:")
    summary = df.groupby("tag")["text_length"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(1).reindex(TAG_ORDER).dropna()
    print(summary.to_string())

    per_jd_tag = (
        df.groupby(["jd_id", "tag"])["text_length"]
          .mean()
          .reset_index()
    )
    print("\nDistribution of per-JD MEAN length (one row per tag):")
    by_tag = per_jd_tag.groupby("tag")["text_length"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).round(1).reindex(TAG_ORDER).dropna()
    by_tag = by_tag.rename(columns={"count": "n_JDs"})
    print(by_tag.to_string())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_segments()
    print(f"Loaded {len(df)} segments across {df['jd_id'].nunique()} JDs")
    print(f"Tag counts: {df['tag'].value_counts().to_dict()}")

    _set_global_style()
    fig_seglen_per_tag(df)
    fig_seglen_per_jd_means(df)
    print_summary(df)


if __name__ == "__main__":
    main()
