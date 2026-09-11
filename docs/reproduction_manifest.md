# Reproduction manifest

| Component | Settings recorded in notebook | Expected saved result |
| --- | --- | --- |
| Synthetic interaction sweep | `SWEEP='lambda'`; 5 lambda values; 20 seeds; 60/40 train/test split | `results_lambda.csv` (100 data rows) |
| Synthetic budget sweep | `SWEEP='beta'`; 5 beta values; 20 seeds; 60/40 train/test split | `results_beta.csv` (100 data rows) |
| Synthetic personalization sweep | `SWEEP='rho'`; 5 rho values; 20 seeds; 60/40 train/test split | `results_rho.csv` (100 data rows) |
| Logged-data OPE | 3 exploration settings; 20 seeds; known logging propensities | `results_observational_ope.csv` (60 data rows) |
| Dunnhumby calibration | 801 eligible households; 5 lambda values; 500 out-of-bag bootstrap replicates | `results_dunnhumby.csv` (2,500 data rows) |

## Model configuration

- Primary synthetic and semi-synthetic outcome models: `GradientBoostingRegressor` with 100 estimators, maximum depth 4, and learning rate 0.05.
- Logged-data nuisance outcome models: XGBoost with 100 trees, maximum depth 3, learning rate 0.05, minimum child weight 10, row and column subsampling 0.90, and histogram tree construction.

## Verification

After execution, compare generated row counts with the table above. The figure notebook reads `results_lambda.csv`, `results_beta.csv`, and `results_rho.csv` and writes the five PNG files included in the repository.

## Remaining release requirements

Before a new TORS submission, record the CPU model, RAM, operating system, Python version, package-lock output, and wall-clock runtime of each notebook from a clean environment. Add the repository release DOI after creating the archival release.
