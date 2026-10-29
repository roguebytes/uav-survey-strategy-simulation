#!/usr/bin/env python3
"""Target density at which the time-based comparison breaks even.

Extends time_cost_model.py: the verification tour is simulated across target
densities at the two survey operating points, and for each time-model variant
the density at which Strategy 2's mission time equals Strategy 1's is found by
linear interpolation between the simulated densities. The path-length model is
included as a check against the crossover of Section 5.4.3 (about 0.48 for the
40 m survey).

Output: time_cost_crossover.json
Usage: /opt/anaconda3/bin/python time_cost_crossover.py [same parameters as time_cost_model.py]
"""
import os, sys, json, time, argparse
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)
import numpy as np
from minefield_util import gen_flag_field_rates, try_gen_minefield
from survey_cost import lawnmower
from time_cost_model import nn_tour, mission, passes, N, R, C1, H_V

__author__ = "Frank Loewenich"

OPERATING_POINTS = [("40 m", 0.004, 40.0 / H_V), ("15 m", 0.089, 15.0 / H_V)]
DENSITIES = [round(x, 2) for x in np.arange(0.02, 0.62, 0.02)]
MODELS = [
    ("path length (thesis)", None, None, None),
    ("time, fixed airspeed, hover-and-capture everywhere, turn at every waypoint", 0, True, "every waypoint"),
    ("time, fixed airspeed, continuous sweeps, turn at every waypoint", 0, False, "every waypoint"),
    ("time, fixed airspeed, continuous sweeps, sharp turns only", 0, False, "sharp"),
    ("time, airspeed ~ altitude, continuous sweeps, turn at every waypoint", 1, False, "every waypoint"),
    ("time, airspeed ~ altitude, continuous sweeps, sharp turns only", 1, False, "sharp"),
    ("time, airspeed ~ altitude, hover-and-capture everywhere, turn at every waypoint", 1, True, "every waypoint"),
]


def zero_crossings(dens, sav):
    """Densities at which the saving changes sign, by linear interpolation."""
    out = []
    for i in range(1, len(dens)):
        a, b = sav[i - 1], sav[i]
        if a == 0:
            out.append(dens[i - 1])
        elif (a > 0) != (b > 0):
            out.append(dens[i - 1] + (dens[i] - dens[i - 1]) * a / (a - b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell", type=float, default=9.0)
    ap.add_argument("--v-v", type=float, default=5.0)
    ap.add_argument("--v-max", type=float, default=12.0)
    ap.add_argument("--t-turn", type=float, default=2.0)
    ap.add_argument("--t-dwell", type=float, default=3.0)
    ap.add_argument("--t-sortie", type=float, default=0.0)
    ap.add_argument("--experiments", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = dict(ell=args.ell, v_v=args.v_v, v_max=args.v_max, t_turn=args.t_turn,
             t_dwell=args.t_dwell, t_sortie=args.t_sortie)
    t_all = time.time()
    out = {"_meta": dict(params=p, R=R, C1=C1, grid=[N, N], h_v=H_V, densities=DENSITIES,
                         experiments=args.experiments, seed=args.seed)}
    for label, fpr, r in OPERATING_POINTS:
        rows = []
        for d in DENSITIES:
            np.random.seed(args.seed)
            L, nf, ns = [], [], []
            for _ in range(args.experiments):
                M = try_gen_minefield(N, N, d)
                F = gen_flag_field_rates(M, R, fpr)
                a, b, c = nn_tour(F)
                L.append(a); nf.append(b); ns.append(c)
            rows.append(dict(density=d, r=r, c_survey=lawnmower(r), tour_length=float(np.mean(L)),
                             flagged=float(np.mean(nf)), sharp_turns=float(np.mean(ns))))
        res = {"fpr": fpr, "r": r, "rows": rows, "crossover": {}, "saving": {}}
        print(f"\n{label} survey (FPR {fpr}, r {r:.2f})")
        for name, alpha, sweep_dwell, vt in MODELS:
            sav = []
            for sc in rows:
                if alpha is None:
                    t1, t2 = float(C1), sc["c_survey"] + sc["tour_length"]
                else:
                    t1, t2 = mission(sc, p, alpha, sweep_dwell, vt)
                sav.append(1 - t2 / t1)
            zc = zero_crossings(DENSITIES, sav)
            res["saving"][name] = dict(zip([str(d) for d in DENSITIES], sav))
            res["crossover"][name] = zc
            print(f"  {name:84s} crossover at density {', '.join(f'{z:.3f}' for z in zc) or 'none in range'}")
        out[label] = res
    out["_meta"]["total_time_s"] = time.time() - t_all
    with open(f"{SIM_DIR}/time_cost_crossover.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved time_cost_crossover.json ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
