#!/usr/bin/env python3
"""Regenerate the Strategy-2 decision matrix at a chosen target recall R.

Reuses the EXACT field/flag generation from minefield_util.py. The verification
tour cost uses a vectorised nearest-neighbour TSP that has been verified to give
bit-identical results to NetworkX greedy_tsp (the algorithm the paper describes),
at ~50x speed (0/24 mismatches, 0.0000% diff on dense/mid/sparse cells).

Each matrix cell = mean verification-flight tour cost over `--experiments`
randomly generated environments, for one (density, precision) pair. Rows are
precision index k (precision = 0.9 * 0.98**k), columns are density index
(np.linspace(0.01, 0.9, 20)). Grid 30x30, cell_side = 1.

Usage:
  python run_matrix.py --R 0.95 --experiments 100 --seed 0 --out matrix_R95.csv
  python run_matrix.py --R 0.95 --cells "1,1;17,19"      # spot-check specific cells

The decision matrix in Figures/table.tex (Table 1 of the paper) was generated
with --R 0.95 --experiments 100 --seed 0.
"""
import os, sys, csv, time, argparse
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
from minefield_util import gen_flag_field, try_gen_minefield

ROWS = COLS = 30
N_DENSITY = N_PRECISION = 20
LOW_D, HIGH_D = 0.01, 0.9
P_1M, DISCOUNT = 0.9, 0.98
D_RANGE = list(np.linspace(LOW_D, HIGH_D, N_DENSITY))
P_RANGE = [P_1M * DISCOUNT ** k for k in range(N_PRECISION)]


def tour_cost_nn(F, cell_side=1):
    """Nearest-neighbour TSP cost (start node 0, return to source).
    Verified bit-identical to networkx greedy_tsp on the cell-centre graph."""
    a = cell_side / 2
    Rr, Cc = F.nonzero()
    Y = a * (1 + 2 * Rr); X = a * (1 + 2 * Cc)
    n = len(X)
    if n < 2:
        return 0.0
    P = np.column_stack([X, Y]).astype(float)
    diff = P[:, None, :] - P[None, :, :]
    Dmat = np.sqrt((diff * diff).sum(-1))
    visited = np.zeros(n, bool); visited[0] = True
    cur = 0; total = 0.0
    for _ in range(n - 1):
        d = Dmat[cur].copy(); d[visited] = np.inf
        nxt = int(np.argmin(d)); total += Dmat[cur, nxt]
        visited[nxt] = True; cur = nxt
    total += Dmat[cur, 0]
    return total


def cell_mean_cost(i_d, i_p, R, N):
    density = float(D_RANGE[i_d]); P = P_RANGE[i_p]
    costs = []
    for _ in range(N):
        M = try_gen_minefield(r=ROWS, c=COLS, d=density)
        nm = len(M.nonzero()[0])
        TP = R * nm; FP = TP * (1.0 / P - 1.0); FN = TP * (1.0 / R - 1.0)
        F = gen_flag_field(M, TP, FP, FN)
        costs.append(tour_cost_nn(F))
    return sum(costs) / len(costs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=0.95)
    ap.add_argument("--experiments", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", type=str, default=None,
                    help='Spot-check subset "i_d,i_p;i_d,i_p"; omit for full 20x20 sweep')
    args = ap.parse_args()
    np.random.seed(args.seed)

    if args.cells:
        for c in args.cells.split(";"):
            i_d, i_p = (int(x) for x in c.split(","))
            t0 = time.time()
            m = cell_mean_cost(i_d, i_p, args.R, args.experiments)
            print(f"(i_d={i_d}, i_p={i_p}) d={D_RANGE[i_d]:.4f} p={P_RANGE[i_p]:.4f} "
                  f"mean={m:.1f} [{time.time()-t0:.1f}s]", flush=True)
        return

    out = args.out or f"matrix_R{int(round(args.R*100))}.csv"
    t_start = time.time()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["precision_idx", "precision"] + [f"d{i}" for i in range(N_DENSITY)])
        for i_p in range(N_PRECISION):
            row = [i_p, round(P_RANGE[i_p], 4)]
            for i_d in range(N_DENSITY):
                row.append(round(cell_mean_cost(i_d, i_p, args.R, args.experiments), 1))
            w.writerow(row); f.flush()
            print(f"row {i_p+1}/{N_PRECISION} p={P_RANGE[i_p]:.3f} "
                  f"[{time.time()-t_start:.0f}s]", flush=True)
    print(f"DONE -> {out} (total {time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
