# Joint Recommendation and Promotion Causal Framework

Reproducibility package for:

**"Quantifying the Revenue Degradation of Decoupled Recommendation 
and Promotion Systems: A Joint Causal Framework"**  
Jagruthi Nalajala — submitted to ACM Transactions on Recommender Systems

---

## Repository Contents

| File | Description |
|------|-------------|
| `nalajala2026_simulation.ipynb` | Primary synthetic simulation (lambda, beta, rho sweeps) |
| `nalajala2026_dunnhumby.ipynb` | Semi-synthetic validation on Dunnhumby dataset |
| `nalajala2026_figures.ipynb` | Generates all 5 paper figures from result CSVs |
| `nalajala2026_observational_ope.ipynb` | Logged-data OPE simulation (DM vs DR evaluation) |
| `results_lambda.csv` | Results for Table 2 (interaction strength sweep) |
| `results_beta.csv` | Results for Table 3 (budget tightness sweep) |
| `results_rho.csv` | Results for Table 4 (personalization sweep) |
| `results_dunnhumby.csv` | Results for Table 7 (Dunnhumby bootstrap) |
| `results_observational_ope.csv` | Results for Table 8 (OPE evaluation) |
| `figure1_gap_curve.png` | Figure 1: Revenue gap curve across targeting fractions |
| `figure2_gap_by_lambda.png` | Figure 2: Revenue gap by interaction strength |
| `figure3_interference_bias.png` | Figure 3: Recommendation-mediated bias |
| `figure4_gap_by_beta.png` | Figure 4: Revenue gap by budget tightness |
| `figure5_gap_by_rho.png` | Figure 5: Revenue gap by personalization level |

---

## Requirements

```
python==3.10
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.3.0
xgboost==1.7.6
matplotlib==3.7.2
seaborn==0.12.2
jupyter==1.0.0
---

## How to Reproduce

1. **Primary simulation** (Tables 2, 3, 4 and Figures 2, 4, 5):  
   Open `nalajala2026_simulation.ipynb`. Set the `SWEEP` variable at the top to `'lambda'`, `'beta'`, or `'rho'` and run all cells.

2. **Dunnhumby validation** (Table 7):  
   Download the Dunnhumby Complete Journey dataset from [dunnhumby.com/source-files](https://www.dunnhumby.com/source-files/). Place all CSV files in the same folder, then run `nalajala2026_dunnhumby.ipynb`.

3. **OPE evaluation** (Table 8):  
   Run `nalajala2026_observational_ope.ipynb` with no additional data required.

4. **Figures**:  
   Run `nalajala2026_figures.ipynb` to regenerate all figures from the result CSVs.

---

## Data Availability

This repository will be made fully public upon paper acceptance.  
The Dunnhumby Complete Journey dataset is publicly available at [dunnhumby.com/source-files](https://www.dunnhumby.com/source-files/).
