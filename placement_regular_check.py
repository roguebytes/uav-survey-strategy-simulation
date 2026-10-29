#!/usr/bin/env python3
"""Regular (lattice) target placement versus independent placement.

Complements clustering_analysis.py at the other extreme: targets on a regular
lattice at the same density, the most even placement possible. For a fixed
number of waypoints a tour is shortest when they are concentrated and longest
when they are spread evenly, so the lattice bounds the verification cost from
above as the Thomas cluster process bounds it from below.

Output: placement_regular_check.json
Usage: /opt/anaconda3/bin/python placement_regular_check.py
"""
import os, sys, json
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
from minefield_util import gen_flag_field_rates, try_gen_minefield
from run_matrix import tour_cost_nn

__author__ = "Frank Loewenich"

N = 30
R = 0.95
N_ENV = 200
SPACINGS = (4, 3, 2)          # lattice spacing in cells: density 1/16, 1/9, 1/4
FPRS = (0.004, 0.089)


def lattice_field(s, offset):
    M = np.zeros((N, N), bool)
    M[offset[0]::s, offset[1]::s] = True
    return M


def main():
    out = {"_meta": {"R": R, "grid": [N, N], "environments": N_ENV}}
    for s in SPACINGS:
        d = 1.0 / (s * s)
        for fpr in FPRS:
            np.random.seed(3)
            lat, ber = [], []
            for k in range(N_ENV):
                off = np.random.randint(0, s, size=2)
                lat.append(tour_cost_nn(gen_flag_field_rates(lattice_field(s, off), R, fpr)))
                ber.append(tour_cost_nn(gen_flag_field_rates(try_gen_minefield(N, N, d), R, fpr)))
            res = dict(density=d, fpr=fpr,
                       lattice_mean=float(np.mean(lat)), bernoulli_mean=float(np.mean(ber)),
                       ratio=float(np.mean(lat) / np.mean(ber)))
            out[f"spacing {s}, fpr {fpr}"] = res
            print(f"spacing {s} (d={d:.4f}) fpr={fpr}: lattice {res['lattice_mean']:.0f} "
                  f"vs bernoulli {res['bernoulli_mean']:.0f} (ratio {res['ratio']:.3f})", flush=True)
    with open(f"{SIM_DIR}/placement_regular_check.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved placement_regular_check.json")


if __name__ == "__main__":
    main()
