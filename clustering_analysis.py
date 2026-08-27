#!/usr/bin/env python3
"""Clustered target placement versus independent placement.

Targets are drawn from a Thomas cluster process (cluster centres scattered
over the grid, each spawning a Poisson-distributed number of targets with
Gaussian scatter) truncated to the grid, with the intensity calibrated so
the realised target density matches the independent Bernoulli placement.
Verification cost is compared between the two placement models at
representative densities, using the measured 40 m operating rates
(R = 0.95, FPR = 0.029).

These results are reported in the Discussion of the paper ("Clustered
Target Distributions").

Output: clustering_analysis.json
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

N = 30


def thomas_field(d, mean_children=5, sigma=1.5, calib=1.0):
    """Thomas process truncated to the grid.

    ``calib`` compensates for duplicate collapse (at most one target per
    cell), so the realised density matches the nominal density d.
    """
    n_expected = d * N * N * calib
    n_parents = max(1, int(round(n_expected / mean_children)))
    M = np.zeros((N, N), dtype=bool)
    px = np.random.uniform(0, N, n_parents)
    py = np.random.uniform(0, N, n_parents)
    for x0, y0 in zip(px, py):
        k = np.random.poisson(mean_children)
        xs = np.clip(np.round(np.random.normal(x0, sigma, k)).astype(int),
                     0, N - 1)
        ys = np.clip(np.round(np.random.normal(y0, sigma, k)).astype(int),
                     0, N - 1)
        M[ys, xs] = True
    return M


def main():
    out = {}
    R, fpr = 0.95, 0.029
    for d in (0.06, 0.20, 0.43):
        res = {}
        # Calibrate the Thomas intensity so the realised density matches
        # the Bernoulli one.
        np.random.seed(5)
        trial = np.mean([thomas_field(d).mean() for _ in range(300)])
        calib = d / trial
        generators = (
            ("bernoulli", lambda: try_gen_minefield(N, N, d)),
            ("thomas", lambda: thomas_field(d, calib=calib)),
        )
        for name, gen in generators:
            np.random.seed(21)
            costs, dens = [], []
            for _ in range(200):
                M = gen()
                dens.append(M.mean())
                costs.append(tour_cost_nn(gen_flag_field_rates(M, R, fpr)))
            c = np.array(costs)
            res[name] = dict(
                mean=float(c.mean()),
                sd=float(c.std(ddof=1)),
                realised_density=float(np.mean(dens)),
            )
        out[str(d)] = res
        print(f"d={d}: bernoulli {res['bernoulli']['mean']:.0f} vs "
              f"thomas {res['thomas']['mean']:.0f} "
              f"(realised d {res['thomas']['realised_density']:.3f})",
              flush=True)

    with open(f"{HERE}/clustering_analysis.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved clustering_analysis.json")


if __name__ == "__main__":
    main()
