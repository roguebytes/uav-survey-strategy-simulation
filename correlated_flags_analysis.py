#!/usr/bin/env python3
"""Spatially correlated false positives versus independent per-cell flags.

A high-altitude survey image spans several verification-altitude grid cells,
so the conditions that make the detector flag one empty cell (glare, shadow,
texture) may flag its neighbours in the same image too. The decision table
draws every empty cell's flag independently (Bernoulli with probability FPR).

This script keeps the per-cell marginal FPR fixed and varies the correlation
within a survey-image footprint. The grid is tiled into b x b blocks, b being
the survey swath in cells (the altitude ratio rounded to a whole number). Each
block draws one shared uniform U_block, each cell its own uniform Z_cell, and
with probability phi the cell uses the shared draw:

    W = U_block with probability phi, otherwise Z_cell
    empty cell flagged  <=>  W <= FPR

phi = 0 is the independent model of the thesis, phi = 1 flags every empty cell
of a footprint together, and P(flag | empty) = FPR for every phi. Target cells
are flagged with probability R as before. The same environments and the same
draws are used for every phi (common random numbers), so differences between
phi values are not sampling noise.

Output: correlated_flags_analysis.json
Usage: /opt/anaconda3/bin/python correlated_flags_analysis.py
"""
import os, sys, json, time
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
from minefield_util import try_gen_minefield
from run_matrix import tour_cost_nn

__author__ = "Frank Loewenich"

N = 30
R = 0.95
N_ENV = 200
CASES = [
    ("40 m operating point, FPR 0.004, footprint 4 x 4 cells", 0.004, 4),
    ("15 m operating point, FPR 0.089, footprint 2 x 2 cells", 0.089, 2),
    ("High false-positive rate 0.20, footprint 4 x 4 cells", 0.20, 4),
    ("High false-positive rate 0.40, footprint 4 x 4 cells", 0.40, 4),
]
DENSITIES = (0.06, 0.20, 0.43)
PHIS = (0.0, 0.5, 1.0)


def block_ids(b):
    """Integer id of the b x b survey-image block containing each cell."""
    idx = np.arange(N) // b
    n_blocks = int(idx.max()) + 1
    return idx[:, None] * n_blocks + idx[None, :]


def main():
    out = {"_meta": {"R": R, "grid": [N, N], "environments": N_ENV,
                     "phis": list(PHIS), "densities": list(DENSITIES)}}
    t0 = time.time()
    for label, fpr, b in CASES:
        ids = block_ids(b)
        n_blocks = int(ids.max()) + 1
        out[label] = {"fpr": fpr, "block_side": b}
        for d in DENSITIES:
            np.random.seed(7)
            rng = np.random.default_rng(11)
            acc = {phi: {"cost": [], "flags": [], "false": []} for phi in PHIS}
            for _ in range(N_ENV):
                M = try_gen_minefield(N, N, d)
                Zt = rng.random(M.shape)          # target-cell draws
                Zc = rng.random(M.shape)          # independent empty-cell draws
                U = rng.random(n_blocks)[ids]     # one shared draw per block
                S = rng.random(M.shape)           # selects shared versus own draw
                for phi in PHIS:
                    W = np.where(S < phi, U, Zc)
                    F = ((Zt <= R) & M) | ((W <= fpr) & ~M)
                    acc[phi]["cost"].append(tour_cost_nn(F))
                    acc[phi]["flags"].append(int(F.sum()))
                    acc[phi]["false"].append(int((F & ~M).sum()))
            res = {}
            base = float(np.mean(acc[0.0]["cost"]))
            for phi in PHIS:
                c = np.array(acc[phi]["cost"])
                res[str(phi)] = dict(
                    mean_cost=float(c.mean()), sd_cost=float(c.std(ddof=1)),
                    mean_flags=float(np.mean(acc[phi]["flags"])),
                    mean_false_flags=float(np.mean(acc[phi]["false"])),
                    cost_ratio_to_independent=float(c.mean() / base) if base else None)
            out[label][str(d)] = res
            print(f"{label} | d={d}: " + ", ".join(
                f"phi={phi}: {res[str(phi)]['mean_cost']:.0f} "
                f"(false flags {res[str(phi)]['mean_false_flags']:.1f})" for phi in PHIS),
                flush=True)
    out["_meta"]["total_time_s"] = time.time() - t0
    with open(f"{SIM_DIR}/correlated_flags_analysis.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved correlated_flags_analysis.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
