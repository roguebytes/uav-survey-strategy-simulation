# UAV Survey Strategy Simulation

Simulation code for the manuscript **"Fly High or Fly Low? Adaptive UAV Search
Strategies for Time-Efficient Aerial Detection"** (Loewenich, Maire, Sandino,
Gonzalez — submitted to *Remote Sensing*, 2026, manuscript under review).

The code estimates the expected cost of a two-stage UAV search strategy — a
rapid high-altitude survey followed by a low-altitude verification flight over
detector-flagged locations — and compares it against a constant low-altitude
survey baseline. Sweeping target density against detector precision produces
the pre-computed decision matrix reported in the paper.

## Contents

| File | Purpose |
|---|---|
| `simulation.py` | Single-configuration simulation with detailed logging |
| `minefield_util.py` | Core model: environment generation, conditional flag-field sampling, TSP tour costing |
| `run_matrix.py` | Regenerates the full decision matrix (Table 2 of the paper) |
| `tsp_solver_comparison.py` | Reproduces the TSP solver comparison (Table 1 of the paper) |
| `tsp_solver_comparison.json` | Results of that comparison as reported |
| `logs/` | Raw experiment logs (JSON) from the reported runs |
| `docs/` | Experiment log spreadsheets |
| `archive/` | Earlier exploratory scripts, kept for provenance |

## Requirements

Python 3.10+, `numpy`, `networkx`, `matplotlib`.

## Reproducing the paper's results

The decision matrix (Table 2), at recall target 0.95 with 100 environments
per cell and a fixed seed:

    python run_matrix.py --R 0.95 --experiments 100 --seed 0

The TSP solver comparison (Table 1):

    python tsp_solver_comparison.py --experiments 100 --seed 0

## Citation

Citation details will be added on publication. Until then, please cite the
manuscript: Loewenich, F.; Maire, F.; Sandino, J.; Gonzalez, F. *Fly High or
Fly Low? Adaptive UAV Search Strategies for Time-Efficient Aerial Detection.*
Submitted to Remote Sensing, 2026.

## License

MIT — see [LICENSE](LICENSE).
