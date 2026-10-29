#!/usr/bin/env python3
"""Crossover density under greedy and Christofides routing at the 40 m operating point.

Complements solver_bias_operating_points.py: identical flag fields are routed
with both solvers at densities spanning the total-cost crossover of the 40 m
survey, and the density at which each solver's saving changes sign is found by
linear interpolation.

Output: solver_bias_crossover.json
Usage: /opt/anaconda3/bin/python solver_bias_crossover.py [--experiments 100] [--seed 0]
"""
import os, sys, json, time, argparse
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
import networkx as nx
from minefield_util import gen_flag_field_rates, try_gen_minefield
from run_matrix import tour_cost_nn
from solver_bias_operating_points import christofides_cost, N, R, C1

__author__ = "Frank Loewenich"

FPR, R_RATIO, C_SURVEY = 0.004, 40.0 / 11.0, 266
DENSITIES = [0.48, 0.52, 0.56, 0.60, 0.64]


def crossing(dens, sav):
    for i in range(1, len(dens)):
        a, b = sav[i - 1], sav[i]
        if (a > 0) != (b > 0):
            return dens[i - 1] + (dens[i] - dens[i - 1]) * a / (a - b)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {"_meta": dict(R=R, C1=C1, grid=[N, N], fpr=FPR, c_survey=C_SURVEY,
                         densities=DENSITIES, experiments=args.experiments, seed=args.seed,
                         networkx=nx.__version__)}
    t_all = time.time()
    rows = []
    for d in DENSITIES:
        np.random.seed(args.seed)
        g, c, n = [], [], []
        t0 = time.time()
        for _ in range(args.experiments):
            M = try_gen_minefield(N, N, d)
            F = gen_flag_field_rates(M, R, FPR)
            n.append(int(F.sum())); g.append(tour_cost_nn(F)); c.append(christofides_cost(F))
        gm, cm = float(np.mean(g)), float(np.mean(c))
        row = dict(density=d, mean_nodes=float(np.mean(n)), greedy=gm, christofides=cm,
                   saving_greedy=1 - (C_SURVEY + gm) / C1, saving_christofides=1 - (C_SURVEY + cm) / C1,
                   time_s=time.time() - t0)
        rows.append(row)
        print(f"density {d:.2f} nodes~{row['mean_nodes']:5.1f} | greedy {gm:6.1f} saving {100*row['saving_greedy']:+5.1f}% "
              f"| christofides {cm:6.1f} saving {100*row['saving_christofides']:+5.1f}% [{row['time_s']:.0f}s]", flush=True)
    out["rows"] = rows
    out["crossover_greedy"] = crossing(DENSITIES, [r["saving_greedy"] for r in rows])
    out["crossover_christofides"] = crossing(DENSITIES, [r["saving_christofides"] for r in rows])
    out["_meta"]["total_time_s"] = time.time() - t_all
    print(f"crossover greedy {out['crossover_greedy']} christofides {out['crossover_christofides']}")
    with open(f"{SIM_DIR}/solver_bias_crossover.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved solver_bias_crossover.json ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
