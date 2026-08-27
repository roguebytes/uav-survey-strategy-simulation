#!/usr/bin/env python3
"""Uncertainty quantification for the decision table.

(a) For the two measured false-positive rates (0.089 at 15 m, 0.029 at 40 m):
    mean, standard deviation, and 95% confidence interval of the
    verification cost at every density, over 400 environments across
    4 seeds.
(b) Convergence: half-width of the 95% confidence interval of a cell mean
    versus the number of environments N, at three representative cells,
    supporting the choice of N = 100 per cell.

These results are reported in the Discussion of the paper ("Sample Size and
Statistical Power").

Output: uncertainty_analysis.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from minefield_util import gen_flag_field_rates, try_gen_minefield  # noqa: E402
from run_matrix import tour_cost_nn  # noqa: E402

__author__ = "Frank Loewenich"


def main():
    densities = np.linspace(0.01, 0.9, 20)
    out = {"rows": {}, "convergence": {}}

    for fpr in (0.089, 0.029):
        rows = []
        for d in densities:
            costs = []
            for seed in range(4):
                np.random.seed(1000 + seed)
                costs += [
                    tour_cost_nn(gen_flag_field_rates(
                        try_gen_minefield(30, 30, float(d)), 0.95, fpr))
                    for _ in range(100)
                ]
            c = np.array(costs)
            rows.append(dict(
                d=float(d),
                mean=float(c.mean()),
                sd=float(c.std(ddof=1)),
                ci95=float(1.96 * c.std(ddof=1) / np.sqrt(len(c))),
            ))
        out["rows"][str(fpr)] = rows
        print(f"fpr={fpr}: done", flush=True)

    for d, fpr in ((0.06, 0.029), (0.43, 0.029), (0.06, 0.089)):
        np.random.seed(7)
        pool = np.array([
            tour_cost_nn(gen_flag_field_rates(
                try_gen_minefield(30, 30, d), 0.95, fpr))
            for _ in range(800)
        ])
        conv = []
        for n in (25, 50, 100, 200, 400, 800):
            sub = pool[:n]
            conv.append(dict(
                N=n,
                mean=float(sub.mean()),
                half=float(1.96 * sub.std(ddof=1) / np.sqrt(n)),
            ))
        out["convergence"][f"d{d}_fpr{fpr}"] = conv
        print(f"convergence d={d} fpr={fpr}: done", flush=True)

    with open(f"{HERE}/uncertainty_analysis.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved uncertainty_analysis.json")


if __name__ == "__main__":
    main()
