#!/usr/bin/env python3
"""Generate the decision table under the (R, FPR) detector parameterisation.

The detector is characterised by two density-independent per-cell
probabilities,

    R   = prob(flag | target cell)   (recall, held at the target)
    FPR = prob(flag | empty cell)    (false-positive rate)

instead of (R, precision). Precision becomes a derived, density-dependent
output, p(d) = R*d / (R*d + FPR*(1 - d)), reported for readability but never
used to drive the simulation. Both inputs are probabilities by construction,
so no derived quantity can exceed 1.

Two row modes are available:

* Geometric FPR sweep (``--fpr-min``/``--fpr-max``): rows are FPR values
  spaced geometrically over the given range. This mode produced the decision
  table in the paper (Table 2).
* Precision-anchored rows (default): each row k corresponds to the nominal
  precision p_k = 0.9 * 0.98**k of the legacy matrix (see run_matrix.py),
  converted to an FPR at the field experiment's own target density rho_exp:
  FPR_k = R * rho_exp * (1/p_k - 1) / (1 - rho_exp). Row k then reproduces
  the measured operating point (R, p_k) exactly at rho_exp and extrapolates
  it correctly at every other density. ``--rho-exp`` is a scientific input
  (the proportion of observation cells containing a target during the field
  trials) and is stamped into the output filename.

Environment generation and tour costing are identical to run_matrix.py
(30x30 grid, 20 densities, 100 environments per cell, nearest-neighbour TSP
verified bit-identical to networkx greedy_tsp).

Usage:
  python run_matrix_fpr.py --R 0.95 --fpr-min 0.01 --fpr-max 0.40 \
      --out matrix_fpr_R95_sweep_001_040.csv
  python run_matrix_fpr.py --R 0.95 --rho-exp 0.2 --cells "1,1;17,19"
"""

import argparse
import csv
import os
import sys
import time

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)

import numpy as np  # noqa: E402

from minefield_util import gen_flag_field_rates, try_gen_minefield  # noqa: E402
from run_matrix import (  # noqa: E402
    COLS,
    D_RANGE,
    N_DENSITY,
    N_PRECISION,
    P_RANGE,
    ROWS,
    tour_cost_nn,
)

__author__ = "Frank Loewenich"


def fpr_from_precision(p, R, rho_exp):
    """Invert the precision formula at the experiment's own density."""
    return R * rho_exp * (1.0 / p - 1.0) / (1.0 - rho_exp)


def derived_precision(R, fpr, d):
    """Density-dependent precision implied by the two per-cell rates."""
    return (R * d) / (R * d + fpr * (1.0 - d))


def cell_mean_cost_rates(i_d, R, fpr, n_experiments):
    """Mean verification-tour cost over n_experiments random environments."""
    density = float(D_RANGE[i_d])
    costs = []
    for _ in range(n_experiments):
        M = try_gen_minefield(r=ROWS, c=COLS, d=density)
        F = gen_flag_field_rates(M, R, fpr)
        costs.append(tour_cost_nn(F))
    return sum(costs) / len(costs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=0.95)
    ap.add_argument("--rho-exp", dest="rho_exp", type=float, default=0.2,
                    help="Field experiment's target density (targets / "
                         "observation cells) for precision-anchored rows.")
    ap.add_argument("--experiments", type=int, default=100)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", type=str, default=None,
                    help='Spot-check subset "i_d,i_p;i_d,i_p"; '
                         'omit for the full sweep')
    ap.add_argument("--fpr-min", dest="fpr_min", type=float, default=None,
                    help="Geometric FPR sweep (with --fpr-max): rows are FPR "
                         "values spaced geometrically, no anchoring.")
    ap.add_argument("--fpr-max", dest="fpr_max", type=float, default=None)
    args = ap.parse_args()
    np.random.seed(args.seed)

    if (args.fpr_min is None) != (args.fpr_max is None):
        ap.error("--fpr-min and --fpr-max must be given together")
    if args.fpr_min is not None:
        ratio = (args.fpr_max / args.fpr_min) ** (1.0 / (N_PRECISION - 1))
        fpr_range = [args.fpr_min * ratio ** k for k in range(N_PRECISION)]
    else:
        fpr_range = [fpr_from_precision(p, args.R, args.rho_exp)
                     for p in P_RANGE]
    assert all(0.0 < f < 1.0 for f in fpr_range), \
        "anchored FPR range leaves (0, 1)"

    if args.cells:
        for c in args.cells.split(";"):
            i_d, i_p = (int(x) for x in c.split(","))
            fpr = fpr_range[i_p]
            t0 = time.time()
            m = cell_mean_cost_rates(i_d, args.R, fpr, args.experiments)
            dp = derived_precision(args.R, fpr, D_RANGE[i_d])
            print(f"(i_d={i_d}, i_p={i_p}) d={D_RANGE[i_d]:.4f} "
                  f"fpr={fpr:.4f} p_nom={P_RANGE[i_p]:.4f} "
                  f"p_derived={dp:.4f} mean={m:.1f} "
                  f"[{time.time() - t0:.1f}s]", flush=True)
        return

    if args.out:
        out = args.out
    elif args.fpr_min is not None:
        out = f"matrix_fpr_R{int(round(args.R * 100))}_sweep.csv"
    else:
        out = (f"matrix_fpr_R{int(round(args.R * 100))}"
               f"_rho{int(round(args.rho_exp * 100))}.csv")

    t_start = time.time()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        hdr2 = ("nominal_precision_at_rho_exp" if args.fpr_min is None
                else "unused")
        w.writerow(["row_idx", hdr2, "fpr"]
                   + [f"d{i}" for i in range(N_DENSITY)])
        for i_p in range(N_PRECISION):
            fpr = fpr_range[i_p]
            row = [i_p,
                   round(P_RANGE[i_p], 4) if args.fpr_min is None else "",
                   round(fpr, 5)]
            for i_d in range(N_DENSITY):
                row.append(round(
                    cell_mean_cost_rates(i_d, args.R, fpr, args.experiments),
                    1))
            w.writerow(row)
            f.flush()
            print(f"row {i_p + 1}/{N_PRECISION} fpr={fpr:.4f} "
                  f"[{time.time() - t_start:.0f}s]", flush=True)
    print(f"DONE -> {out} (total {time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
