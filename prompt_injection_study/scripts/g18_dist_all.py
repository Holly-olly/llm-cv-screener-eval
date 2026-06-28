#!/usr/bin/env python3
"""Sandbox g18 — score distributions for all injection conditions on ONE chart.

KDE of the black-box (recruiter persona) fit score over the 50 main JDs for cv_primary:
  Baseline (clean CV)  +  injections A / B / C / D (each appended at CV bottom).
Shows how each injection style shifts the whole score distribution. Neural Haze palette,
dark background, presentation only.

Reads:  results/injection_all_cv_primary.csv
Writes: results/dist_injections_all_cv_primary.png
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "granular_l3_sandbox" / "results" / "injection_all_cv_primary.csv"
FIG = ROOT / "granular_l3_sandbox" / "results" / "dist_injections_all_cv_primary.png"

# Neural Haze
VOID, DEEP, GHOST, SLATE = "#0B0A14", "#1C1928", "#F0EFF8", "#6E6A88"
BASE_C = "#F0EFF8"   # ghostWhite  — clean baseline
A_C    = "#00F5D4"   # neuralCyan
B_C    = "#FF2D78"   # hotMagenta  (strongest)
C_C    = "#9B30FF"   # electricViolet
D_C    = "#C8F135"   # acidLime


def kde(s, grid):
    s = np.asarray(s, float)
    bw = 1.06 * s.std(ddof=1) * len(s) ** (-1 / 5)
    return np.exp(-0.5 * ((grid[:, None] - s[None, :]) / bw) ** 2).sum(1) / (len(s) * bw * np.sqrt(2 * np.pi))


def main():
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

    def col(k):
        return np.array([float(r[k]) for r in rows if r.get(k) not in ("", "None", None)])

    base = col("recruiter_100")
    base_mean = base.mean()
    series = [
        ("Baseline · clean CV",       base,            BASE_C),
        ("D · parser metadata",       col("injD_100"), D_C),
        ("A · “ignore instructions”", col("injA_100"), A_C),
        ("C · rubric satisfied",      col("injC_100"), C_C),
        ("B · fake system notice",    col("injB_100"), B_C),
    ]

    grid = np.linspace(0, 100, 320)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.patch.set_facecolor(VOID); ax.set_facecolor(VOID)

    for lab, vals, c in series:
        if len(vals) < 2:
            continue
        y = kde(vals, grid)
        is_base = (c == BASE_C)
        ax.plot(grid, y, color=c, lw=3.2 if not is_base else 2.4,
                ls="--" if is_base else "-", alpha=0.95, zorder=3)
        ax.fill_between(grid, y, color=c, alpha=0.10 if not is_base else 0.06, zorder=2)
        m = vals.mean()
        ax.axvline(m, color=c, lw=1.0, alpha=0.55, ls=":", zorder=1)
        dlt = m - base_mean
        tag = f"mean {m:.0f}" + ("" if is_base else f"  (Δ{dlt:+.0f})")
        series_label = f"{lab}   —  {tag}"
        ax.plot([], [], color=c, lw=3.2, ls="--" if is_base else "-", label=series_label)

    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Fit score (0–100), black-box recruiter persona", color=GHOST, fontsize=12)
    ax.set_ylabel("Density", color=GHOST, fontsize=12)
    ax.set_title("Prompt injection shifts the score distribution\n"
                 "cv_primary × 50 JDs · same persona, injection appended at CV bottom",
                 color=GHOST, fontsize=15, pad=14, loc="left")
    ax.tick_params(axis="x", colors=GHOST, labelsize=11)
    ax.tick_params(axis="y", colors=SLATE, labelsize=10)
    for sp in ax.spines.values():
        sp.set_color(DEEP)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=DEEP, lw=0.6, alpha=0.5)

    leg = ax.legend(loc="upper right", frameon=False, fontsize=11, labelcolor=GHOST,
                    handlelength=1.6, borderaxespad=0.6)
    fig.tight_layout()
    fig.savefig(FIG, dpi=200, facecolor=VOID)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
