#!/usr/bin/env python3
"""Sandbox g23 — report figures for the prompt-injection study (Neural Haze, dark bg).

fig1 : Gemini L1 mean Δ per condition  vs  Level-3 glass-box (≈0, structural immunity)
fig2 : Gemini L1  vs  OpenAI L1 (gpt-4o-mini) mean Δ per condition
fig3 : noise control on B — per-JD clean μ vs B μ with within-condition SD bands

Reads:  results/master_injection_scores_cv_primary.csv, results/noise_runs_cv_primary.csv
Writes: results/fig1_gemini_vs_glassbox.png, fig2_gemini_vs_openai.png, fig3_noise_B.png
"""
from __future__ import annotations
from pathlib import Path
import csv
import statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
MASTER = R / "master_injection_scores_cv_primary.csv"
NOISE = R / "noise_runs_cv_primary.csv"

# Neural Haze
VOID, DEEP, GHOST, SLATE = "#0B0A14", "#1C1928", "#F0EFF8", "#6E6A88"
VIOLET, CYAN, LIME, MAGENTA, WHITE = "#9B30FF", "#00F5D4", "#C8F135", "#FF2D78", "#F0EFF8"
CONDS = ["A", "B", "C", "D"]
LABELS = {"A": "A · override", "B": "B · system notice",
          "C": "C · rubric", "D": "D · metadata"}


def _num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _style(ax, title, ylabel):
    ax.set_facecolor(VOID)
    ax.set_title(title, color=GHOST, fontsize=14, pad=12, loc="left")
    ax.set_ylabel(ylabel, color=GHOST, fontsize=11)
    ax.tick_params(colors=GHOST, labelsize=10)
    for s in ax.spines.values():
        s.set_color(DEEP)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=DEEP, lw=0.6, alpha=0.6)
    ax.axhline(0, color=SLATE, lw=1.0)


def mean_delta(rows, key):
    v = [_num(r[key]) for r in rows if _num(r.get(key)) is not None]
    return sum(v) / len(v) if v else float("nan")


def fig1_glassbox(rows):
    gem = [mean_delta(rows, f"gem_d{c}") for c in CONDS]
    glass = [0.0 for _ in CONDS]  # structural: injection tagged `other`, dropped pre-scoring
    x = np.arange(len(CONDS)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.6)); fig.patch.set_facecolor(VOID)
    b1 = ax.bar(x - w / 2, gem, w, color=VIOLET, label="Black-box L1 (Gemini)")
    ax.bar(x + w / 2, glass, w, color=LIME, label="Glass-box L3 (structured)", edgecolor=LIME)
    for rect, val in zip(b1, gem):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.3, f"+{val:.1f}",
                ha="center", color=VIOLET, fontsize=10, fontweight="bold")
    ax.text(len(CONDS) - 1 + w / 2, 0.6, "0 — injection dropped\nat segmentation (tag=other)",
            ha="center", color=LIME, fontsize=9)
    _style(ax, "Black-box inflates, glass-box does not", "mean score shift Δ (0–100)")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], color=GHOST)
    ax.legend(frameon=False, labelcolor=GHOST, fontsize=10, loc="upper left")
    fig.tight_layout(); out = R / "fig1_gemini_vs_glassbox.png"
    fig.savefig(out, dpi=200, facecolor=VOID); print("wrote", out.name)


def fig2_providers(rows):
    gem = [mean_delta(rows, f"gem_d{c}") for c in CONDS]
    oai = [mean_delta(rows, f"oai_d{c}") for c in CONDS]
    x = np.arange(len(CONDS)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.6)); fig.patch.set_facecolor(VOID)
    b1 = ax.bar(x - w / 2, gem, w, color=VIOLET, label="Gemini 3.1 Flash Lite")
    b2 = ax.bar(x + w / 2, oai, w, color=CYAN, label="OpenAI gpt-4o-mini")
    for rects, vals, col in [(b1, gem, VIOLET), (b2, oai, CYAN)]:
        for rect, val in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.3, f"+{val:.1f}",
                    ha="center", color=col, fontsize=9, fontweight="bold")
    _style(ax, "Injection effect across providers (Level-1 holistic)", "mean score shift Δ (0–100)")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[c] for c in CONDS], color=GHOST)
    ax.legend(frameon=False, labelcolor=GHOST, fontsize=10, loc="upper right")
    fig.tight_layout(); out = R / "fig2_gemini_vs_openai.png"
    fig.savefig(out, dpi=200, facecolor=VOID); print("wrote", out.name)


def fig3_noise():
    rows = list(csv.DictReader(open(NOISE, encoding="utf-8")))
    jds = []
    for r in rows:
        if r["jd_id"] not in jds:
            jds.append(r["jd_id"])

    def cell(j, c):
        return [int(r["score_100"]) for r in rows
                if r["jd_id"] == j and r["cond"] == c and r["score_100"] not in ("", "None")]
    jds = sorted(jds, key=lambda j: st.mean(cell(j, "clean")) if cell(j, "clean") else 0)
    cl_m = [st.mean(cell(j, "clean")) for j in jds]
    b_m = [st.mean(cell(j, "B")) for j in jds]
    b_sd = [st.pstdev(cell(j, "B")) if len(cell(j, "B")) > 1 else 0 for j in jds]
    cl_sd = [st.pstdev(cell(j, "clean")) if len(cell(j, "clean")) > 1 else 0 for j in jds]

    x = np.arange(len(jds))
    fig, ax = plt.subplots(figsize=(10, 5.8)); fig.patch.set_facecolor(VOID)
    ax.errorbar(x, cl_m, yerr=cl_sd, fmt="o-", color=WHITE, lw=2, capsize=4,
                label="clean (5 runs)")
    ax.errorbar(x, b_m, yerr=b_sd, fmt="o-", color=MAGENTA, lw=2.4, capsize=4,
                label="B · system notice (5 runs)")
    for xi, c, b in zip(x, cl_m, b_m):
        ax.annotate(f"+{b - c:.0f}", (xi, b + 2), color=MAGENTA, fontsize=9, ha="center")
    _style(ax, "Noise control on B: effect (+16.9) ≈ 4.3× run-to-run SD (3.9)",
           "fit score (0–100), mean ± SD over 5 runs")
    short = [j[:16] for j in jds]
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=35, ha="right", color=GHOST, fontsize=8)
    ax.legend(frameon=False, labelcolor=GHOST, fontsize=10, loc="upper left")
    ax.set_ylim(0, 105)
    fig.tight_layout(); out = R / "fig3_noise_B.png"
    fig.savefig(out, dpi=200, facecolor=VOID); print("wrote", out.name)


def main():
    rows = list(csv.DictReader(open(MASTER, encoding="utf-8")))
    fig1_glassbox(rows)
    fig2_providers(rows)
    fig3_noise()


if __name__ == "__main__":
    main()
