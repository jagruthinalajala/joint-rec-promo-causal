#!/usr/bin/env bash
set -euo pipefail

# Pre-specified factual-outcome sensitivity study for the TORS revision.
# Run from the repository root after installing requirements.txt.

mkdir -p results/revised_grid

for n_train in 1500 6000 12000; do
  for lambda in 0.00 0.25 0.50 0.75 1.00; do
    python experiments/run_logged_policy_learning.py \
      --seeds 20 \
      --n-train "$n_train" \
      --n-test 4000 \
      --lambda "$lambda" \
      --exploration 0.25 \
      --budget 0.50 \
      --output "results/revised_grid/logged_n${n_train}_lambda${lambda}.csv"
  done
done
