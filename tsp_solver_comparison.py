#!/usr/bin/env python3
"""Compare NetworkX TSP solvers on the verification-flight graphs.

Justifies the paper's choice of the greedy nearest-neighbour heuristic by
comparing it against Christofides' 1.5-approximation at the four corners of the
density-precision sweep.

Reuses the EXACT field/flag generation from minefield_util (R=0.95, 30x30 grid,
seed 0, 100 experiments), so the greedy nearest-neighbour numbers are consistent
with the paper's decision matrix -- run_matrix.py uses a vectorised
nearest-neighbour TSP that is bit-identical to networkx greedy_tsp.

For each environment the same flag field (hence the same complete graph on the
flagged cell centres) is passed to BOTH solvers, so the cost/time comparison is
like-for-like. In networkx 3.x, traveling_salesman_problem() defaults to
Christofides for undirected graphs, so it is not reported as a separate solver.

Usage:
  /opt/anaconda3/bin/python tsp_solver_comparison.py [--experiments 100] [--seed 0]
"""
import os, sys, time, argparse, json
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
import networkx as nx
from networkx.algorithms.approximation.traveling_salesman import greedy_tsp, christofides
from minefield_util import gen_flag_field, try_gen_minefield

ROWS = COLS = 30
D_RANGE = list(np.linspace(0.01, 0.9, 20))
P_RANGE = [0.9 * 0.98 ** k for k in range(20)]
R = 0.95

LOW_D, HIGH_D = D_RANGE[0], D_RANGE[19]     # 0.01, 0.9
HIGH_P, LOW_P = P_RANGE[0], P_RANGE[19]     # 0.90, ~0.6133

CORNERS = [
    ("Low density, low precision",   LOW_D,  LOW_P),
    ("Low density, high precision",  LOW_D,  HIGH_P),
    ("High density, low precision",  HIGH_D, LOW_P),
    ("High density, high precision", HIGH_D, HIGH_P),
]


def build_graph(F, cell_side=1):
    """Complete graph on flagged cell centres, Euclidean edge weights.
    Equivalent to minefield_util.get_tour's graph, built vectorised."""
    a = cell_side / 2
    Rr, Cc = F.nonzero()
    Y = a * (1 + 2 * Rr); X = a * (1 + 2 * Cc)
    n = len(X)
    Pos = np.column_stack([X, Y]).astype(float)
    diff = Pos[:, None, :] - Pos[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    iu, ju = np.triu_indices(n, k=1)
    G.add_weighted_edges_from(
        (int(i), int(j), float(D[i, j])) for i, j in zip(iu.tolist(), ju.tolist()))
    return G, D


def tour_cost(D, path):
    return float(sum(D[path[k], path[k + 1]] for k in range(len(path) - 1)))


def gen_F(density, P):
    M = try_gen_minefield(r=ROWS, c=COLS, d=density)
    nm = len(M.nonzero()[0])
    TP = R * nm; FP = TP * (1.0 / P - 1.0); FN = TP * (1.0 / R - 1.0)
    return gen_flag_field(M, TP, FP, FN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default=os.path.join(SIM_DIR, "tsp_solver_comparison.json"))
    args = ap.parse_args()
    np.random.seed(args.seed)
    N = args.experiments

    solvers = {
        "greedy":       lambda G: greedy_tsp(G, weight="weight"),
        "christofides": lambda G: christofides(G, weight="weight"),
    }
    results = {}
    t_all = time.time()
    for name, d, p in CORNERS:
        acc = {s: {"cost": [], "time": 0.0} for s in solvers}
        n_nodes = []
        for _ in range(N):
            F = gen_F(d, p)
            G, D = build_graph(F)
            n = G.number_of_nodes()
            n_nodes.append(n)
            for s, fn in solvers.items():
                # Match run_matrix.py's convention for degenerate tours; a solver
                # is only invoked when there are >=3 flagged cells to route.
                if n < 2:
                    cost, dt = 0.0, 0.0
                elif n == 2:
                    cost, dt = 2.0 * float(D[0, 1]), 0.0
                else:
                    t0 = time.time()
                    path = fn(G)
                    dt = time.time() - t0
                    cost = tour_cost(D, path)
                acc[s]["time"] += dt
                acc[s]["cost"].append(cost)
        results[name] = {
            "density": d, "precision": p, "mean_nodes": float(np.mean(n_nodes)),
            **{s: {"mean_cost": float(np.mean(acc[s]["cost"])),
                   "total_time_s": acc[s]["time"]} for s in solvers},
        }
        r = results[name]
        print(f"{name:31s} nodes~{r['mean_nodes']:6.1f} | "
              f"greedy cost={r['greedy']['mean_cost']:8.2f} t={r['greedy']['total_time_s']:7.2f}s | "
              f"christofides cost={r['christofides']['mean_cost']:8.2f} "
              f"t={r['christofides']['total_time_s']:8.2f}s", flush=True)
    results["_meta"] = {"R": R, "grid": [ROWS, COLS], "experiments": N,
                        "seed": args.seed, "networkx": nx.__version__,
                        "total_time_s": time.time() - t_all}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDONE in {time.time() - t_all:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
