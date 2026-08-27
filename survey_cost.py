#!/usr/bin/env python3
"""Survey-leg costs from lawnmower geometry, and the sensitivity figure.

The grid cell side is 1 unit (the verification-altitude camera footprint).
A lawnmower sweep of an N x N grid with a swath of ``sw`` cells costs

    C_lawnmower(sw) = ceil(N/sw) * (N - 1) + (ceil(N/sw) - 1) * sw

which gives the Strategy-1 sweep C1 = 899 units at swath 1 (N = 30) and the
Strategy-2 survey legs at wider swaths. The break-even survey-to-verification
altitude ratio for a verification cost V is

    path-length model:  r* = 1 / (1 - V/C1)
    time model (speed proportional to altitude):  r* = sqrt(1 / (1 - V/C1))

Reads the decision-table sweep (matrix_fpr_R95_sweep_001_040.csv) and plots
the total-mission saving of Strategy 2 against the altitude ratio for a range
of target densities, under both cost models (Figure 6 of the paper).

Output: fig_survey_cost_sensitivity.png
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__author__ = "Frank Loewenich"

HERE = os.path.dirname(os.path.abspath(__file__))
N = 30
BLUE, VERM, GREEN, ORANGE, PINK = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7")


def lawnmower(sw):
    """Lawnmower sweep cost of the N x N grid at swath width sw (cells)."""
    passes = int(np.ceil(N / sw))
    return passes * (N - 1) + (passes - 1) * sw


def main():
    c1 = lawnmower(1.0)
    densities = np.linspace(0.01, 0.9, 20)
    with open(f"{HERE}/matrix_fpr_R95_sweep_001_040.csv") as f:
        rows = list(csv.reader(f))[1:]
    matrix = np.array([[float(x) for x in r[3:]] for r in rows])
    v = matrix[11]  # measured operating row, FPR 0.085

    ratios = np.linspace(1.0, 4.0, 200)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    picks = [(1, BLUE), (4, GREEN), (9, ORANGE), (12, VERM), (17, PINK)]
    panels = ((axes[0], 1, "Path-length model (fixed airspeed)"),
              (axes[1], 2, "Time model (speed proportional to altitude)"))
    for ax, model, title in panels:
        for i_d, col in picks:
            saving = 1 - (1 / ratios ** model) - v[i_d] / c1
            ax.plot(ratios, saving * 100, color=col, lw=2.2,
                    label=f"$\\rho$ = {densities[i_d]:.2f}")
        ax.axhline(0, color="black", lw=1)
        ax.axvline(15 / 11, color="grey", ls=":", lw=1.8)
        ax.text(15 / 11 + 0.04, -52, "this platform\n(15 m / 11 m)",
                fontsize=10, color="grey")
        ax.set_title(title, fontsize=12.5)
        ax.set_xlabel("Survey-to-verification altitude ratio $h_s/h_v$",
                      fontsize=12)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Total-mission saving of Strategy 2 (%)", fontsize=12)
    axes[0].set_ylim(-60, 80)
    axes[0].legend(fontsize=10.5, loc="upper left", title="Target density",
                   title_fontsize=10.5)
    fig.tight_layout()
    fig.savefig(f"{HERE}/fig_survey_cost_sensitivity.png", dpi=300,
                bbox_inches="tight")
    print("saved fig_survey_cost_sensitivity.png")


if __name__ == "__main__":
    main()
