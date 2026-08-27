# UAV Survey Strategy Simulation

Simulation code for the manuscript **"Fly High or Fly Low? Selecting
Time-Efficient UAV Search Strategies for High-Recall Aerial Detection"** (Loewenich, Maire, Sandino,
Gonzalez; submitted to *Remote Sensing*, 2026, manuscript under review).

The code estimates the expected cost of a two-stage UAV search strategy (a
rapid high-altitude survey followed by a low-altitude verification flight over
detector-flagged locations) and compares it against a constant low-altitude
survey baseline on **total mission cost** (survey leg plus verification
flight). The detector is characterised by two density-independent per-cell
probabilities, recall and false-positive rate, with precision a derived
output. Sweeping target density against false-positive rate produces the
pre-computed decision table reported in the paper, and the derived break-even
altitude ratio r* = 1/(1 − V/C₁) generalises the comparison to any platform.

## Contents

| File | Purpose |
|---|---|
| `minefield_util.py` | Core model: environment generation, flag-field sampling (precision-based and rate-based), TSP tour costing |
| `run_matrix_fpr.py` | Generates the decision table under the (R, FPR) parameterisation (Table 3 of the paper) |
| `matrix_fpr_R95_sweep_001_040.csv` | The decision-table sweep as reported (R = 0.95, FPR 0.01-0.40 geometric) |
| `survey_cost.py` | Lawnmower survey-leg costs and the total-mission sensitivity figure (Figure 6) |
| `breakeven_map.py` | Universal break-even map r*(ρ, FPR) (Figure 7) |
| `uncertainty_analysis.py` | Cell-mean confidence intervals and the N = 100 convergence analysis |
| `uncertainty_analysis.json` | Results of that analysis as reported |
| `clustering_analysis.py` | Thomas cluster process vs independent placement |
| `clustering_analysis.json` | Results of that comparison as reported |
| `tsp_solver_comparison.py` | Reproduces the TSP solver comparison (Table 2 of the paper) |
| `tsp_solver_comparison.json` | Results of that comparison as reported |

Retained from the original submission, for provenance:

| File | Purpose |
|---|---|
| `run_matrix.py` | The original (density, precision) decision matrix; also provides the shared tour-costing routine |
| `simulation.py` | Single-configuration simulation with detailed logging |
| `logs/` | Raw experiment logs (JSON) from the original runs |
| `docs/` | Experiment log spreadsheets |
| `archive/` | Earlier exploratory scripts |

## Requirements

Python 3.10+, `numpy`, `networkx`, `matplotlib`.

## Reproducing the paper's results

The decision table (Table 3), at recall target 0.95 with 100 environments
per cell and a fixed seed:

    python run_matrix_fpr.py --R 0.95 --fpr-min 0.01 --fpr-max 0.40 \
        --seed 0 --out matrix_fpr_R95_sweep_001_040.csv

The TSP solver comparison (Table 2):

    python tsp_solver_comparison.py --experiments 100 --seed 0

The survey-cost sensitivity figure (Figure 6) and the break-even map
(Figure 7), both reading the decision-table sweep above:

    python survey_cost.py
    python breakeven_map.py

The uncertainty and convergence analysis, and the clustered-placement
comparison (both reported in the Discussion):

    python uncertainty_analysis.py
    python clustering_analysis.py

All scripts are deterministic under the seeds shown and reproduce the
reported values exactly.

## Citation

Citation details will be added on publication. Until then, please cite the
manuscript: Loewenich, F.; Maire, F.; Sandino, J.; Gonzalez, F. *Fly High or
Fly Low? Selecting Time-Efficient UAV Search Strategies for High-Recall Aerial
Detection.*
Submitted to Remote Sensing, 2026.

## License

MIT; see [LICENSE](LICENSE).
