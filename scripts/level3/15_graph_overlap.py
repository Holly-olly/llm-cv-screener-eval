#!/usr/bin/env python3
"""
Level 3 — Step 15: Bipartite graph visualisation of CV ↔ JD overlap.

Simplest useful version (no GNN, no deep graph learning) — just a static
bipartite NetworkX graph that visualises how embedding-based similarity
matches CV segments to JD segments.

This script draws ONE figure with N panels side-by-side, one panel per
(CV, JD) pair in the PAIRS list. Useful for visually comparing a strong
match against a weak match.

Method (per pair):
  - Read the pre-built cosine matrix (script 12).
  - For every JD node, draw an edge to its TOP-K best CV matches by
    cosine similarity. The top-1 edge is bolder; top-2 / top-3 are
    thinner and more transparent.
  - Edges below MIN_SIM are dropped entirely (declutter).

Reads:
  - results/level3/similarity_matrices/{cv_id}__vs__{jd_id}.npz

Writes:
  - results/level3/figures/fig_graph_comparison.png

Run:
    cd ~/Documents/cool-cohen/llm_evaluation
    python3 scripts/level3/15_graph_overlap.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from palette import (
    VOID, GHOST, MIST,
    CYAN, VIOLET, MAGENTA, ACID_LIME,
)

MAT_DIR = ROOT / "results" / "level3" / "similarity_matrices"
FIG_DIR = ROOT / "results" / "level3" / "figures"

# Pairs to draw side-by-side. Order = left → right in the figure.
PAIRS: list[tuple[str, str, str]] = [
    ("cv_primary", "maki_people_senior_psychometrician", "STRONG MATCH"),
    ("cv_primary", "ecs_web_software_developer_5c342c",  "WEAK MATCH"),
]

TOP_K   = 3
MIN_SIM = 0.30

RANK_STYLE = {
    1: {"width": 2.6, "alpha": 0.95},
    2: {"width": 1.4, "alpha": 0.55},
    3: {"width": 0.9, "alpha": 0.35},
}

TAG_COLORS = {
    "skills":     CYAN,
    "experience": VIOLET,
    "mixed":      MAGENTA,
    "education":  ACID_LIME,
}

MAX_LABEL_CHARS = 36


def short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_LABEL_CHARS else text[: MAX_LABEL_CHARS - 1] + "…"


def build_graph(sim, cv_line_ids, cv_tags, cv_texts,
                jd_line_ids, jd_tags, jd_texts) -> tuple[nx.Graph, int]:
    G = nx.Graph()
    n_cv, n_jd = sim.shape
    for i in range(n_cv):
        G.add_node(f"cv_{i}", bipartite=0, tag=str(cv_tags[i]),
                   label=short(str(cv_texts[i])), line_id=int(cv_line_ids[i]))
    for j in range(n_jd):
        G.add_node(f"jd_{j}", bipartite=1, tag=str(jd_tags[j]),
                   label=short(str(jd_texts[j])), line_id=int(jd_line_ids[j]))

    n_edges = 0
    for j in range(n_jd):
        col = sim[:, j]
        order = np.argsort(-col)
        kept = 0
        for i in order:
            s = float(col[i])
            if s < MIN_SIM:
                break
            G.add_edge(f"cv_{i}", f"jd_{j}", weight=s, rank=kept + 1)
            kept += 1
            n_edges += 1
            if kept == TOP_K:
                break
    return G, n_edges


def draw_pair(ax, cv_id: str, jd_id: str, title: str) -> None:
    npz_path = MAT_DIR / f"{cv_id}__vs__{jd_id}.npz"
    if not npz_path.exists():
        ax.text(0.5, 0.5, f"matrix not found:\n{npz_path.name}",
                color=GHOST, ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    data = np.load(npz_path, allow_pickle=True)
    sim         = data["sim"]
    cv_line_ids = data["cv_line_ids"]
    cv_tags     = data["cv_tags"]
    cv_texts    = data["cv_texts"]
    jd_line_ids = data["jd_line_ids"]
    jd_tags     = data["jd_tags"]
    jd_texts    = data["jd_texts"]

    G, n_edges = build_graph(sim, cv_line_ids, cv_tags, cv_texts,
                             jd_line_ids, jd_tags, jd_texts)

    # Order nodes by tag for visual clustering
    tag_order = ["skills", "experience", "mixed", "education"]
    def sort_key(node):
        t = G.nodes[node]["tag"]
        return (tag_order.index(t) if t in tag_order else 99, G.nodes[node]["line_id"])

    cv_nodes = sorted([n for n, d in G.nodes(data=True) if d["bipartite"] == 0], key=sort_key)
    jd_nodes = sorted([n for n, d in G.nodes(data=True) if d["bipartite"] == 1], key=sort_key)

    def positions(nodes, x):
        ys = np.linspace(1.0, 0.0, len(nodes))
        return {n: (x, ys[i]) for i, n in enumerate(nodes)}

    pos = {**positions(cv_nodes, x=0.0), **positions(jd_nodes, x=1.0)}

    ax.set_facecolor(VOID)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    # Headers per side
    ax.text(0.0, 1.04, f"CV  ({cv_id})", color=GHOST, ha="center",
            va="bottom", fontsize=10, fontweight="bold")
    ax.text(1.0, 1.04, f"JD  ({jd_id[:40]}{'…' if len(jd_id) > 40 else ''})",
            color=GHOST, ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Edges
    for u, v, d in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        style = RANK_STYLE.get(d["rank"], RANK_STYLE[3])
        jd_node = v if v.startswith("jd_") else u
        edge_color = TAG_COLORS.get(G.nodes[jd_node]["tag"], MIST)
        ax.plot([x1, x2], [y1, y2],
                color=edge_color, linewidth=style["width"],
                alpha=style["alpha"], zorder=1)

    # Nodes
    for node, (x, y) in pos.items():
        tag   = G.nodes[node]["tag"]
        color = TAG_COLORS.get(tag, MIST)
        deg   = G.degree[node]
        size  = 50 + 14 * deg
        ax.scatter([x], [y], s=size, color=color, edgecolors=GHOST,
                   linewidths=0.5, zorder=3)
        is_cv = node.startswith("cv_")
        label = f"[{G.nodes[node]['line_id']:>2}] {G.nodes[node]['label']}"
        ax.text(x + (-0.02 if is_cv else 0.02), y, label,
                color=GHOST, fontsize=6.4, va="center",
                ha="right" if is_cv else "left", zorder=4)

    # Subtitle with headline numbers
    max_sim = float(sim.max())
    fit_A   = float(sim.max(axis=0).mean())
    sub = (f"global MaxSim = {fit_A:.3f}    max cell = {max_sim:.3f}    "
           f"edges drawn = {n_edges}    top-{TOP_K} per JD    sim ≥ {MIN_SIM}")
    ax.set_title(f"{title}\n{sub}",
                 color=GHOST, fontsize=11, fontweight="bold", pad=24)


def draw_legend(ax) -> None:
    """Single shared legend in its own slim axis."""
    ax.set_facecolor(VOID)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    line_h = 0.07
    y = 0.96
    ax.text(0.05, y, "tags (node colour)", color=GHOST,
            fontsize=10, va="center", fontweight="bold")
    for t in ["skills", "experience", "mixed", "education"]:
        y -= line_h
        ax.scatter([0.10], [y], s=80,
                   color=TAG_COLORS[t], edgecolors=GHOST, linewidths=0.5)
        ax.text(0.18, y, t, color=GHOST, fontsize=9, va="center")

    y -= line_h * 1.5
    ax.text(0.05, y, "edge rank (per JD node)", color=GHOST,
            fontsize=10, va="center", fontweight="bold")
    for rk, st in RANK_STYLE.items():
        y -= line_h
        ax.plot([0.05, 0.18], [y, y],
                color=MIST, linewidth=st["width"], alpha=st["alpha"])
        ax.text(0.21, y, f"top-{rk}", color=GHOST, fontsize=9, va="center")

    y -= line_h * 1.5
    ax.text(0.05, y, "node size ∝ degree", color=MIST,
            fontsize=9, va="center", fontstyle="italic")


def main() -> None:
    n = len(PAIRS)
    # Layout: one column per pair + one slim legend column
    fig = plt.figure(figsize=(8.0 * n + 2.4, 11), facecolor=VOID)
    widths = [8.0] * n + [2.4]
    gs = fig.add_gridspec(1, n + 1, width_ratios=widths, wspace=0.20)

    for k, (cv_id, jd_id, title) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[0, k])
        draw_pair(ax, cv_id, jd_id, title)

    legend_ax = fig.add_subplot(gs[0, n])
    draw_legend(legend_ax)

    fig.suptitle("CV ↔ JD overlap — bipartite top-K matches by cosine similarity",
                 color=GHOST, fontsize=13, fontweight="bold", y=0.995)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_graph_comparison.png"
    fig.savefig(out, dpi=170, facecolor=VOID, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
