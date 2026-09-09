#!/usr/bin/env python3
"""Grid-size robustness of V/C1 (backs the universality claim of Section 4.3).

For square grids of side N in {20, 30, 45}, compute V (mean verification-tour
cost over 100 environments, seed 0) and C1 = C(1) at representative
(rho, FPR) pairs, and report the drift of V/C1 relative to the N = 30 value.
Output: grid_size_check.json
"""
import numpy as np, json, math, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from minefield_util import gen_flag_field_rates, try_gen_minefield
from run_matrix import tour_cost_nn

def C1(N):
    return math.ceil(N / 1) * (N - 1) + (math.ceil(N / 1) - 1) * 1

PAIRS = [(0.06, 0.010), (0.06, 0.085), (0.43, 0.085), (0.81, 0.010)]
SIZES = (20, 30, 45)
out = {"pairs": {}, "max_drift_pct": 0.0}
for (d, fpr) in PAIRS:
    ratios = {}
    for N in SIZES:
        np.random.seed(0)
        costs = [tour_cost_nn(gen_flag_field_rates(try_gen_minefield(N, N, d), 0.95, fpr)) for _ in range(100)]
        ratios[N] = float(np.mean(costs)) / C1(N)
    base = ratios[30]
    drift = {N: abs(ratios[N] - base) / base * 100.0 for N in SIZES}
    out["pairs"][f"d{d}_fpr{fpr}"] = {"V_over_C1": ratios, "drift_pct_vs_N30": drift}
    out["max_drift_pct"] = max(out["max_drift_pct"], max(drift.values()))
    print(f"d={d} fpr={fpr}: " + ", ".join(f"N={N}: {ratios[N]:.4f} ({drift[N]:.1f}%)" for N in SIZES), flush=True)

json.dump(out, open(f"{HERE}/grid_size_check.json", "w"), indent=1)
print(f"max drift vs N=30: {out['max_drift_pct']:.1f}%")
