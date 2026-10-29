#!/usr/bin/env python3
"""Solver-induced bias at the measured operating points.

Table 4.3 bounds the greedy nearest-neighbour tour against Christofides'
algorithm at the four corners of the decision table. This script routes
identical flag fields with both solvers at the four scenario rows of Table 5.3
and totals each Strategy-2 mission against the 899-unit Strategy-1 sweep, so
the heuristic's effect on the strategy verdict is explicit.

Output: solver_bias_operating_points.json
Usage: /opt/anaconda3/bin/python solver_bias_operating_points.py [--experiments 200] [--seed 0]
"""
import os, sys, json, time, argparse
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
import networkx as nx
from networkx.algorithms.approximation.traveling_salesman import christofides
from minefield_util import gen_flag_field_rates, try_gen_minefield
from run_matrix import tour_cost_nn
from tsp_solver_comparison import build_graph, tour_cost

__author__ = "Frank Loewenich"

N = 30
R = 0.95
C1 = 899
SCENARIOS = [
    ("Sparse, 40 m", 0.06, 0.004, 266),
    ("Sparse, 15 m", 0.06, 0.089, 659),
    ("Near break-even, 40 m", 0.43, 0.004, 266),
    ("Dense, 40 m", 0.81, 0.004, 266),
]


def christofides_cost(F):
    G, D = build_graph(F)
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    if n == 2:
        return 2.0 * float(D[0, 1])
    return tour_cost(D, christofides(G, weight="weight"))


def totals(c_verify, c_survey):
    c2 = c_survey + c_verify
    return dict(c_verify=c_verify, c2=c2, saving=1 - c2 / C1,
                r_star=1 / (1 - c_verify / C1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {"_meta": {"R": R, "C1": C1, "grid": [N, N], "experiments": args.experiments,
                     "seed": args.seed, "networkx": nx.__version__}}
    t_all = time.time()
    for label, d, fpr, c_survey in SCENARIOS:
        np.random.seed(args.seed)
        greedy, chris, nodes = [], [], []
        t0 = time.time()
        for _ in range(args.experiments):
            M = try_gen_minefield(N, N, d)
            F = gen_flag_field_rates(M, R, fpr)
            nodes.append(int(F.sum()))
            greedy.append(tour_cost_nn(F))
            chris.append(christofides_cost(F))
        g, c = float(np.mean(greedy)), float(np.mean(chris))
        res = dict(density=d, fpr=fpr, c_survey=c_survey, mean_nodes=float(np.mean(nodes)),
                   greedy=totals(g, c_survey), christofides=totals(c, c_survey),
                   greedy_over_christofides=(g / c) if c else None,
                   saving_points_christofides_minus_greedy=100 * (g - c) / C1,
                   time_s=time.time() - t0)
        out[label] = res
        print(f"{label:22s} nodes~{res['mean_nodes']:5.1f} | greedy {g:6.1f} saving "
              f"{100 * res['greedy']['saving']:+5.1f}% | christofides {c:6.1f} saving "
              f"{100 * res['christofides']['saving']:+5.1f}% | ratio {res['greedy_over_christofides']:.3f} "
              f"[{res['time_s']:.0f}s]", flush=True)
    out["_meta"]["total_time_s"] = time.time() - t_all
    with open(f"{SIM_DIR}/solver_bias_operating_points.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved solver_bias_operating_points.json ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
