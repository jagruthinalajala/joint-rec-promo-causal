# Dunnhumby Complete Journey setup

## Data access

Obtain the Complete Journey source files from Dunnhumby under the provider's applicable terms. This repository does not redistribute the dataset.

## Required source files

Place the following files in one local data directory:

- `transaction_data.csv`
- `hh_demographic.csv`
- `coupon.csv`
- `coupon_redempt.csv`
- `product.csv`
- `campaign_desc.csv`

Run the Dunnhumby notebook from that directory, or update its documented data-directory variable before execution. The analysis reads only these public source files and writes `results_dunnhumby.csv`.

## Important interpretation

The notebook derives covariates and spending summaries from the source data, then simulates potential outcomes under the specified response surface. It is a semi-synthetic calibration study, not a causal analysis of the observed Dunnhumby transactions.
