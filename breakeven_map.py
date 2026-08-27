#!/usr/bin/env python3
"""Universal break-even map r*(rho, FPR) = 1/(1 - V/C1), from the sweep CSV.

V/C1 is dimensionless and grid-size-insensitive (2.5-4.4% drift over
N = 20/30/45), so the requirement surface is platform-independent; a platform
contributes only its achievable (FPR, r) operating line. The figure overlays
the two measured operating points reported in the paper (Figure 7).

Output: fig_breakeven_map.png
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__author__ = "Frank Loewenich"

HERE = os.path.dirname(os.path.abspath(__file__))
C1 = 899.0
BLUE, ORANGE = "#0072B2", "#E69F00"


def main():
    densities = np.linspace(0.01, 0.9, 20)
    with open(f"{HERE}/matrix_fpr_R95_sweep_001_040.csv") as f:
        rows = list(csv.reader(f))[1:]
    v = np.array([[float(x) for x in r[3:]] for r in rows])
    fpr = np.array([float(r[2]) for r in rows])
    frac = v / C1
    r_star = np.where(frac < 1.0,
                      1.0 / (1.0 - np.clip(frac, None, 0.999999)),
                      np.inf)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    levels = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    finite = np.where(np.isfinite(r_star), r_star, np.nan)
    cf = ax.contourf(densities, fpr, finite, levels=levels,
                     cmap=plt.cm.viridis_r, extend="max")
    cs = ax.contour(densities, fpr, finite, levels=levels, colors="white",
                    linewidths=1.0)
    ax.clabel(cs, fmt=lambda x: f"$r^*$ = {x:g}", fontsize=9.5)
    ax.text(0.775, 0.20, "$r^*$ > 10:\nbeyond practical\nplatforms",
            color="white", fontsize=10, ha="center", va="center")

    ax.axhline(0.089, color=ORANGE, ls="--", lw=2.0)
    ax.text(0.015, 0.097,
            "15 m survey: measured FPR 0.089, available $r$ = 1.36 "
            "($< r^*$ everywhere)",
            fontsize=9.5, color=ORANGE, va="bottom")
    ax.axhline(0.029, color=BLUE, ls="--", lw=2.0)
    ax.text(0.015, 0.0315,
            "40 m survey: measured FPR 0.029, available $r$ = 3.64",
            fontsize=9.5, color=BLUE, va="bottom")
    ax.plot([0.44], [0.029], marker="o", ms=9, color=BLUE, zorder=5,
            markerfacecolor="white", markeredgewidth=2.2)
    ax.annotate("crossover\n$\\rho \\approx 0.44$", (0.44, 0.029),
                textcoords="offset points", xytext=(14, -34), fontsize=9.5,
                color=BLUE)

    ax.set_yscale("log")
    ax.set_ylim(fpr[0], fpr[-1])
    ax.set_xlim(0.01, 0.9)
    yticks = [0.01, 0.02, 0.03, 0.05, 0.09, 0.15, 0.25, 0.40]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:g}" for t in yticks])
    ax.minorticks_off()
    ax.set_xlabel("Target density $\\rho$", fontsize=12.5)
    ax.set_ylabel("False-positive rate $\\mathrm{FPR}$ (survey altitude)",
                  fontsize=12.5)
    cb = fig.colorbar(cf, ax=ax, pad=0.015)
    cb.set_label(
        "Required survey-to-verification altitude ratio $r^*$",
        fontsize=11.5)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_breakeven_map.png", dpi=300,
                bbox_inches="tight")
    print("saved fig_breakeven_map.png")


if __name__ == "__main__":
    main()
