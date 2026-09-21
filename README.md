# Joint Recommendation and Promotion Causal Framework

Reproducibility package for the manuscript:

> Jagruthi Nalajala. *Quantifying the Revenue Degradation of Decoupled Recommendation and Promotion Systems: A Joint Causal Framework.*

## Scope

The experiments use (1) fully synthetic potential-outcomes simulations, (2) a semi-synthetic calibration based on public Dunnhumby Complete Journey data, and (3) a fully synthetic logged-data policy-evaluation simulation. No proprietary retailer data are included or required.

The Dunnhumby analysis does **not** estimate a causal effect from observed transactions. It uses observed household covariates and spending patterns to calibrate a simulated response surface; see the manuscript and notebook for details.

## Repository map

| Manuscript result | Source | Saved output |
| --- | --- | --- |
| Interaction, budget, and personalization sweeps (Tables 2--4; Figures 2, 4, and 5) | `nalajala2026_simulation.ipynb` | `results_lambda.csv`, `results_beta.csv`, `results_rho.csv` |
| Semi-synthetic Dunnhumby calibration (Table 7) | `nalajala2026_dunnhumby.ipynb` | `results_dunnhumby.csv` |
| Logged-data direct-method and doubly robust evaluation (Table 8) | `nalajala2026_observational_ope.ipynb` | `results_observational_ope.csv` |
| Factual-outcome policy-learning comparison (Table 9) | `experiments/run_logged_policy_learning.py` and `experiments/run_revised_grid.sh` | `results/revised_grid/revised_grid_summary.csv` and the 15 condition-level CSV files in `results/revised_grid/` |
| All manuscript figures | `nalajala2026_figures.ipynb` | `figure1_gap_curve.png` through `figure5_gap_by_rho.png` |

The committed CSVs and PNGs are the outputs used to prepare the manuscript. They allow the figures and summary tables to be inspected without rerunning the simulations.

## Environment

Create an isolated Python 3.10 environment and install the pinned dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The notebooks are intended to run on CPU. No GPU is required. The first full synthetic sweep may take substantial time because it evaluates 20 independent seeds at five parameter values.

## Reproduction order

1. Run `nalajala2026_simulation.ipynb` once for each `SWEEP` setting: `lambda`, `beta`, and `rho`.
2. Run `nalajala2026_observational_ope.ipynb`.
3. Follow `docs/dunnhumby_data_setup.md`, then run `nalajala2026_dunnhumby.ipynb`.
4. Run `nalajala2026_figures.ipynb` after the three synthetic result CSVs are present.

The revised learned-policy sensitivity study is run separately with:

```bash
bash experiments/run_revised_grid.sh
```

Its pre-specified design and interpretation rule are documented in `docs/revised_empirical_plan.md`.

For a first check, inspect the committed CSVs and run the figure notebook. Exact parameter grids, seeds, model settings, output names, and expected row counts are recorded in `docs/reproduction_manifest.md`.

## Data availability

The Dunnhumby Complete Journey source files must be obtained by each user from the data provider under its terms. They are not redistributed in this repository. The required filenames and directory layout are documented in `docs/dunnhumby_data_setup.md`.

## Reproducibility notes

- All simulation seeds, split rules, parameter grids, and model hyperparameters are visible in the notebooks and summarized in the reproduction manifest.
- The primary experiments use 20 independent simulation seeds; reported synthetic intervals are standard deviations across seeds, not confidence intervals.
- The Dunnhumby calibration uses 500 out-of-bag bootstrap resamples of 801 eligible households.
- The factual-outcome comparison reports approximate 95% confidence intervals for pairwise policy-value differences across 20 independent simulation replicates. Results should be interpreted conditionally: joint learning can outperform the decoupled learner under sufficiently strong interactions, but it is not uniformly superior to the pooled additive baseline.

## Citation

See `CITATION.cff` for citation metadata. A release DOI should be added here after the repository is archived.
