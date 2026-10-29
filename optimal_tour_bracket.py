#!/usr/bin/env python3
"""Bracket the optimal verification tour at the four operating points.

Section 5.6.6 compares the greedy nearest-neighbour tour with Christofides'
tour, but neither is the optimum. Over the same flag fields (seed 0, 200
environments) this script computes, per environment, the greedy tour, the
Christofides tour, a 2-opt local search started from each, and the Held-Karp
1-tree lower bound (subgradient ascent on node penalties). The optimal tour
lies between the Held-Karp bound and the best 2-opt tour, so the greedy
tour's excess over the optimum is stated as an interval, and each Strategy-2
total is restated with the bracket.

Output: optimal_tour_bracket.json
Usage: /opt/anaconda3/bin/python optimal_tour_bracket.py [--experiments 200] [--seed 0] [--iters 200]
"""
import os, sys, json, time, argparse
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
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


def nn_tour(D):
    n = D.shape[0]
    vis = np.zeros(n, bool); vis[0] = True
    cur = 0; t = [0]
    for _ in range(n - 1):
        d = D[cur].copy(); d[vis] = np.inf
        cur = int(np.argmin(d)); vis[cur] = True; t.append(cur)
    return t


def two_opt(D, tour):
    """2-opt local search, best move per position, until no improving move remains."""
    t = np.array(tour); n = len(t)
    improved = True
    while improved:
        improved = False
        for i in range(n - 2):
            a, b = t[i], t[i + 1]
            j = np.arange(i + 2, n)
            c = t[j]; d = t[(j + 1) % n]
            gain = D[a, b] + D[c, d] - D[a, c] - D[b, d]
            k = int(np.argmax(gain))
            if gain[k] > 1e-9:
                jj = j[k]
                t[i + 1:jj + 1] = t[i + 1:jj + 1][::-1]
                improved = True
    return t.tolist()


def closed_cost(D, t):
    return tour_cost(D, list(t) + [t[0]])


def one_tree(W):
    n = W.shape[0]
    T = minimum_spanning_tree(W[1:, 1:]).tocoo()
    deg = np.zeros(n, int)
    np.add.at(deg, T.row + 1, 1); np.add.at(deg, T.col + 1, 1)
    w = float(T.data.sum())
    two = np.argsort(W[0, 1:])[:2] + 1
    w += float(W[0, two].sum())
    deg[0] = 2; deg[two] += 1
    return w, deg


def held_karp(D, ub, iters=200):
    """Held-Karp lower bound by subgradient ascent (Held, Wolfe and Crowder schedule)."""
    n = D.shape[0]
    if n < 3:
        return 2.0 * float(D[0, 1]) if n == 2 else 0.0
    pi = np.zeros(n); best = -np.inf; lam = 2.0; stall = 0
    off = 1 - np.eye(n)
    for _ in range(iters):
        W = D + pi[:, None] + pi[None, :]
        shift = -W.min() + 1.0
        w, deg = one_tree((W + shift) * off)
        L = w - n * shift - 2.0 * pi.sum()
        if L > best + 1e-9:
            best = L; stall = 0
        else:
            stall += 1
            if stall >= 10:
                lam *= 0.5; stall = 0
        g = deg - 2
        gn = float((g * g).sum())
        if gn == 0 or lam < 1e-4:
            break
        pi = pi + lam * (ub - L) / gn * g
    return float(best)


def totals(c_verify, c_survey):
    c2 = c_survey + c_verify
    return dict(c_verify=c_verify, c2=c2, saving=1 - c2 / C1, r_star=1 / (1 - c_verify / C1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    out = {"_meta": {"R": R, "C1": C1, "grid": [N, N], "experiments": args.experiments,
                     "seed": args.seed, "hk_iters": args.iters}}
    t_all = time.time()
    for label, d, fpr, c_survey in SCENARIOS:
        np.random.seed(args.seed)
        rows = []
        t0 = time.time()
        for _ in range(args.experiments):
            M = try_gen_minefield(N, N, d)
            F = gen_flag_field_rates(M, R, fpr)
            n = int(F.sum())
            g = tour_cost_nn(F)
            G, D = build_graph(F)
            if n < 3:
                c = g; g2 = g; c2 = g; h = g
            else:
                ct = christofides(G, weight="weight")
                c = tour_cost(D, ct)
                g2 = closed_cost(D, two_opt(D, nn_tour(D)))
                c2 = closed_cost(D, two_opt(D, ct[:-1]))
                h = held_karp(D, min(g2, c2), args.iters)
            rows.append((n, g, c, g2, c2, min(g2, c2), h))
        A = np.array(rows, float)
        n_, g, c, g2, c2, b, h = A.mean(axis=0)
        res = dict(density=d, fpr=fpr, c_survey=c_survey, mean_nodes=float(n_),
                   greedy=totals(float(g), c_survey), christofides=totals(float(c), c_survey),
                   two_opt_from_greedy=totals(float(g2), c_survey),
                   two_opt_from_christofides=totals(float(c2), c_survey),
                   best_two_opt=totals(float(b), c_survey), held_karp=totals(float(h), c_survey),
                   ratios=dict(greedy_over_christofides=float(g / c), greedy_over_best=float(g / b),
                               greedy_over_hk=float(g / h), christofides_over_hk=float(c / h),
                               best_over_hk=float(b / h)),
                   greedy_excess_over_optimum_pct=[100 * float(g / b - 1), 100 * float(g / h - 1)],
                   saving_points_optimum_minus_greedy=[100 * float(g - b) / C1, 100 * float(g - h) / C1],
                   per_env_sd=dict(greedy=float(A[:, 1].std(ddof=1)), best=float(A[:, 5].std(ddof=1)),
                                   hk=float(A[:, 6].std(ddof=1))),
                   time_s=time.time() - t0)
        out[label] = res
        print(f"{label:22s} n~{n_:5.1f} | greedy {g:6.1f} | chris {c:6.1f} | 2opt(g) {g2:6.1f} | "
              f"2opt(c) {c2:6.1f} | best {b:6.1f} | HK {h:6.1f} | g/best {g/b:.3f} g/HK {g/h:.3f} "
              f"best/HK {b/h:.3f} [{res['time_s']:.0f}s]", flush=True)
    out["_meta"]["total_time_s"] = time.time() - t_all
    here = os.path.dirname(os.path.abspath(__file__))
    with open(f"{here}/optimal_tour_bracket.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved optimal_tour_bracket.json ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
