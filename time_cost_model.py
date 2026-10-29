#!/usr/bin/env python3
"""Time-based mission cost with airspeed, turn and dwell effects.

The decision table and Tables 5.2 and 5.3 cost both strategies in path length.
This script converts the scenarios of Table 5.3 to mission time under

    T = L * ell / v(h) + n_turns * t_turn + n_images * t_dwell

summed over the flights of each strategy, with

    v(h) = min(v_v * (h / h_v) ** alpha, v_max)

alpha = 0 is a fixed airspeed, alpha = 1 an airspeed proportional to altitude
(the motion-blur and overlap limit of Section 5.5.3).

Strategy 1, sweep at h_v:          L = C1, passes = N, turns = N - 1, images = N^2
Strategy 2, survey at h_s = r h_v: L = C(r), passes = ceil(N / r), turns = passes - 1,
                                   images = passes^2
Strategy 2, verification at h_v:   L = tour length, images = flagged cells,
                                   turns = every waypoint, or only the sharp turns
                                   (heading change over 45 degrees) along the tour

Two capture protocols are reported: hover-and-capture on every flight (as flown
in this thesis, so the sweeps dwell at every image) and continuous capture on
the two lawnmower sweeps with dwell only at the verification waypoints (the
examiner's framing, and common survey practice).

Output: time_cost_model.json and a printed table.
Usage: /opt/anaconda3/bin/python time_cost_model.py [--ell 9] [--v-v 5] [--v-max 12]
       [--t-turn 2] [--t-dwell 3] [--t-sortie 0] [--experiments 200] [--seed 0]
"""
import os, sys, json, math, time, argparse
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
from minefield_util import gen_flag_field_rates, try_gen_minefield
from survey_cost import lawnmower

__author__ = "Frank Loewenich"

N = 30
R = 0.95
C1 = 899
H_V = 11.0
SCENARIOS = [
    ("Sparse, 40 m", 0.06, 0.004, 40.0 / H_V),
    ("Sparse, 15 m", 0.06, 0.089, 15.0 / H_V),
    ("Near break-even, 40 m", 0.43, 0.004, 40.0 / H_V),
    ("Dense, 40 m", 0.81, 0.004, 40.0 / H_V),
]
DWELLS = (0.0, 1.0, 2.0, 3.0, 5.0)
TURNS = (0.0, 1.0, 2.0, 4.0)


def passes(r):
    return math.ceil(round(N / r, 9))


def nn_tour(F):
    """Nearest-neighbour tour as in run_matrix.tour_cost_nn: length, waypoints, sharp turns."""
    a = 0.5
    Rr, Cc = F.nonzero()
    P = np.column_stack([a * (1 + 2 * Cc), a * (1 + 2 * Rr)]).astype(float)
    n = len(P)
    if n < 2:
        return 0.0, n, 0
    if n == 2:
        return 2.0 * float(np.linalg.norm(P[0] - P[1])), n, 2
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    visited = np.zeros(n, bool); visited[0] = True
    order = [0]; cur = 0; total = 0.0
    for _ in range(n - 1):
        d = D[cur].copy(); d[visited] = np.inf
        nxt = int(np.argmin(d)); total += D[cur, nxt]
        visited[nxt] = True; cur = nxt; order.append(nxt)
    total += D[cur, 0]
    order.append(0)
    pts = P[order]
    seg = np.diff(pts, axis=0)                      # n legs of the closed tour
    inc = np.vstack([seg[-1:], seg[:-1]])           # incoming leg at each waypoint
    cos = (inc * seg).sum(1) / (np.linalg.norm(inc, axis=1) * np.linalg.norm(seg, axis=1))
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return float(total), n, int((ang > 45.0).sum())


def flight_time(L, h_ratio, turns, images, p, alpha, dwell):
    v = min(p["v_v"] * h_ratio ** alpha, p["v_max"])
    return L * p["ell"] / v + turns * p["t_turn"] + (images * p["t_dwell"] if dwell else 0.0)


def mission(sc, p, alpha, sweep_dwell, verification_turns):
    """Return (T1, T2) in seconds for one scenario under one model."""
    r, L_v, n_f, n_sharp = sc["r"], sc["tour_length"], sc["flagged"], sc["sharp_turns"]
    t1 = flight_time(C1, 1.0, N - 1, N * N, p, alpha, sweep_dwell)
    k = passes(r)
    t_survey = flight_time(lawnmower(r), r, k - 1, k * k, p, alpha, sweep_dwell)
    n_turn = n_f if verification_turns == "every waypoint" else n_sharp
    t_verify = flight_time(L_v, 1.0, n_turn, n_f, p, alpha, True)
    return t1, t_survey + t_verify + p["t_sortie"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell", type=float, default=9.0, help="cell side in metres")
    ap.add_argument("--v-v", type=float, default=5.0, help="airspeed at the verification altitude, m/s")
    ap.add_argument("--v-max", type=float, default=12.0, help="platform speed cap, m/s")
    ap.add_argument("--t-turn", type=float, default=2.0, help="seconds per turn")
    ap.add_argument("--t-dwell", type=float, default=3.0, help="seconds per hover-and-capture image")
    ap.add_argument("--t-sortie", type=float, default=0.0, help="extra launch and recovery for Strategy 2, s")
    ap.add_argument("--experiments", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = dict(ell=args.ell, v_v=args.v_v, v_max=args.v_max, t_turn=args.t_turn,
             t_dwell=args.t_dwell, t_sortie=args.t_sortie)

    # 1. Simulated verification tours at the scenario rows (same generator as Table 5.3).
    scen = []
    for label, d, fpr, r in SCENARIOS:
        np.random.seed(args.seed)
        L, nf, ns = [], [], []
        for _ in range(args.experiments):
            M = try_gen_minefield(N, N, d)
            F = gen_flag_field_rates(M, R, fpr)
            a, b, c = nn_tour(F)
            L.append(a); nf.append(b); ns.append(c)
        scen.append(dict(label=label, density=d, fpr=fpr, r=r, c_survey=lawnmower(r),
                         passes=passes(r), survey_images=passes(r) ** 2,
                         tour_length=float(np.mean(L)), flagged=float(np.mean(nf)),
                         sharp_turns=float(np.mean(ns))))

    # 2. Main table at the chosen parameters.
    models = [
        ("path length (thesis)", None, None, None),
        ("time, fixed airspeed, hover-and-capture everywhere, turn at every waypoint", 0, True, "every waypoint"),
        ("time, fixed airspeed, continuous sweeps, turn at every waypoint", 0, False, "every waypoint"),
        ("time, fixed airspeed, continuous sweeps, sharp turns only", 0, False, "sharp"),
        ("time, airspeed ~ altitude, continuous sweeps, turn at every waypoint", 1, False, "every waypoint"),
        ("time, airspeed ~ altitude, continuous sweeps, sharp turns only", 1, False, "sharp"),
        ("time, airspeed ~ altitude, hover-and-capture everywhere, turn at every waypoint", 1, True, "every waypoint"),
    ]
    table = {}
    print(f"params: {p}\n")
    hdr = f"{'model':84s}" + "".join(f"{s['label']:>24s}" for s in scen)
    print(hdr)
    for name, alpha, sweep_dwell, vt in models:
        row = {}
        for s in scen:
            if alpha is None:
                t1, t2 = float(C1), s["c_survey"] + s["tour_length"]
            else:
                t1, t2 = mission(s, p, alpha, sweep_dwell, vt)
            row[s["label"]] = dict(T1=t1, T2=t2, saving=1 - t2 / t1)
        table[name] = row
        print(f"{name:84s}" + "".join(f"{100 * row[s['label']]['saving']:+22.1f} %" for s in scen))

    # 3. Sensitivity of the saving to dwell and turn time (continuous sweeps, every waypoint a turn).
    sens = {}
    for alpha in (0, 1):
        for s in scen:
            grid = {}
            for td in DWELLS:
                for tt in TURNS:
                    q = dict(p, t_dwell=td, t_turn=tt)
                    t1, t2 = mission(s, q, alpha, False, "every waypoint")
                    grid[f"dwell={td},turn={tt}"] = 1 - t2 / t1
            sens[f"alpha={alpha}|{s['label']}"] = grid
    print("\nscenario summary:")
    for s in scen:
        print(f"  {s['label']:22s} r={s['r']:.2f} C_survey={s['c_survey']:.0f} passes={s['passes']} "
              f"survey images={s['survey_images']} tour={s['tour_length']:.0f} "
              f"flagged={s['flagged']:.1f} sharp turns={s['sharp_turns']:.1f}")
    out = dict(_meta=dict(params=p, R=R, C1=C1, grid=[N, N], h_v=H_V,
                          experiments=args.experiments, seed=args.seed),
               scenarios=scen, table=table, sensitivity=sens)
    with open(f"{SIM_DIR}/time_cost_model.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved time_cost_model.json")


if __name__ == "__main__":
    main()
