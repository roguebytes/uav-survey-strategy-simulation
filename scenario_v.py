#!/usr/bin/env python3
"""Scenario-table verification costs at a measured false-positive rate.

Computes V(rho) for the scenario table (Table 4) at the measured FPR of an
operating point, using the same environment/flag/tour primitives as the
decision table, plus the fixed start cell of the tour model (the flag field
always contains the depot cell (0,0), matching the closed tour that begins
at a fixed cell; Section 3.1). 200 environments per density, seed 0, RNG
continuous across the density sweep. This recipe reproduces the published
15 m-detector scenario values (V = 337 at rho 0.057, FPR 0.089) and the
legacy 40 m values to within seed noise.

Usage: python scenario_v.py <FPR> [n_env]
Prints V at every density plus the interpolated total-cost crossover
(V = C1 - Cs = 899 - 266 for the 40 m survey).
"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from minefield_util import gen_flag_field_rates, try_gen_minefield
from run_matrix import tour_cost_nn

FPR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0045
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
C1, CS40 = 899, 266

D = np.linspace(0.01, 0.9, 20)
np.random.seed(0)
means = []
for d in D:
    costs = []
    for _ in range(N):
        F = gen_flag_field_rates(try_gen_minefield(30, 30, float(d)), 0.95, FPR)
        F[0][0] = 1
        costs.append(tour_cost_nn(F))
    m = sum(costs) / len(costs)
    means.append(m)
    print(f"rho={d:.4f}  V={m:.1f}", flush=True)

margin = C1 - CS40
cross = None
for i in range(1, len(D)):
    if (means[i - 1] - margin) * (means[i] - margin) <= 0:
        t = (margin - means[i - 1]) / (means[i] - means[i - 1])
        cross = D[i - 1] + t * (D[i] - D[i - 1])
        break
print(f"FPR={FPR}  n_env={N}  crossover rho (V = {margin}): {cross:.3f}" if cross
      else f"FPR={FPR}  n_env={N}  no crossover in range")
