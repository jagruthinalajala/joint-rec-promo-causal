# Revised empirical plan: factual logged-outcome learning

This plan replaces the original learned-policy experiment, which trained on all simulated potential outcomes per training customer. In the revised design, each training customer has one logged action and one observed outcome. Complete potential outcomes are retained only by the simulator for held-out evaluation.

## Research question

Under what interaction strength and logged-data coverage does a joint policy learner outperform a decoupled policy learner?

## Fixed design

- Action space: 10 recommendation actions by 3 offer actions (30 joint actions).
- Train/test sizes: 1,500 / 4,000; 6,000 / 4,000; and 12,000 / 4,000.
- Interaction strengths: lambda = 0.00, 0.25, 0.50, 0.75, 1.00.
- Logging policy: stochastic mixture with exploration epsilon = 0.25 and known propensities.
- Budget fraction: 0.50.
- Replicates: 20 independent random seeds per condition.
- Evaluation: true incremental revenue on an independent held-out simulation sample, using hidden potential outcomes only after policy learning.

## Compared policies

1. **Joint cell learner:** one factual-outcome model per joint action.
2. **Decoupled marginal learner:** separate factual-outcome models by recommendation and offer margin; their independently selected actions are combined.
3. **Pooled additive learner:** a pooled action-feature ridge baseline that does not model product-offer interaction.

## Reported quantities

- Mean incremental revenue per customer for each policy.
- Pairwise value differences: joint minus decoupled and joint minus pooled.
- Standard deviation across seeds and 95% normal-approximation intervals for the simulation mean.
- Minimum logged propensity and action-cell count diagnostics.

## Interpretation rule

The revised paper will claim a practical learned-policy advantage only in conditions where the confidence interval for the relevant pairwise value difference excludes zero. Otherwise, the result will be reported as inconclusive or as evidence of the finite-sample cost of joint learning.
