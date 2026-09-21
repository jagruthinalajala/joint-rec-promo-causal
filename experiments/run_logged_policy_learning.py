"""Factual-outcome policy-learning simulation for the TORS revision.

Training sees exactly one logged action and one observed outcome per customer.
The complete potential-outcome tensor is retained only by the simulator and is
used on an independent test sample to evaluate the policies' true value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


R, O, D = 10, 3, 10
BASELINE_ACTION = 0


def make_population(n: int, seed: int):
    rng = np.random.RandomState(seed)
    x = rng.multivariate_normal(np.zeros(D), np.eye(D), size=n)
    affinity = softmax(x @ rng.normal(size=(D, R)), axis=1)
    price_sensitivity = expit(x @ rng.normal(size=D))
    baseline = np.abs(x @ rng.normal(size=(D, R)) + rng.normal(size=R)) + 10
    coupling = rng.normal(size=(R, O))
    coupling -= coupling.mean(axis=1, keepdims=True)
    return x, affinity, price_sensitivity, baseline, coupling


def potential_outcomes(affinity, price_sensitivity, baseline, coupling, lam, seed):
    """Simulator-only potential outcomes; never passed to a training routine."""
    rng = np.random.RandomState(seed)
    n = len(affinity)
    discounts = np.array([0.0, 0.10, 0.25])
    y = np.empty((n, R, O))
    for r in range(R):
        for o in range(O):
            main = baseline[:, r] + price_sensitivity * discounts[o] * baseline.mean(axis=1)
            interaction = affinity[:, r] * baseline[:, r] * coupling[r, o] * price_sensitivity
            y[:, r, o] = np.maximum(main + lam * interaction + rng.normal(0, 1.5, n), 0.0)
    return y


def sample_logged_actions(affinity, price_sensitivity, exploration, seed):
    """Known stochastic logging policy with positive probability for all actions."""
    rng = np.random.RandomState(seed)
    rec_score = affinity
    offer_score = np.column_stack((
        0.35 * (1 - price_sensitivity),
        0.65 * price_sensitivity,
        1.00 * price_sensitivity,
    ))
    logits = (rec_score[:, :, None] + offer_score[:, None, :]).reshape(len(affinity), -1)
    exploit = softmax(2.0 * logits, axis=1)
    propensity = (1 - exploration) * exploit + exploration / (R * O)
    uniforms = rng.uniform(size=len(affinity))
    actions = (uniforms[:, None] > np.cumsum(propensity, axis=1)).sum(axis=1)
    return actions, propensity


def observed_outcomes(y, actions):
    flat = y.reshape(len(y), R * O)
    return flat[np.arange(len(y)), actions]


def make_gbr(seed):
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_leaf=10, random_state=seed,
    )


def fit_joint_cells(x, actions, y, seed):
    models = []
    for action in range(R * O):
        rows = np.flatnonzero(actions == action)
        if len(rows) < 20:
            raise RuntimeError(f"Insufficient overlap for action {action}: {len(rows)} factual rows")
        models.append(make_gbr(seed + action).fit(x[rows], y[rows]))
    return models


def joint_scores(models, x):
    return np.column_stack([model.predict(x) for model in models])


def fit_decoupled_marginals(x, actions, y, seed):
    rec = []
    offer = []
    action_r, action_o = actions // O, actions % O
    for r in range(R):
        rows = np.flatnonzero(action_r == r)
        rec.append(make_gbr(seed + r).fit(x[rows], y[rows]))
    for o in range(O):
        rows = np.flatnonzero(action_o == o)
        offer.append(make_gbr(seed + 100 + o).fit(x[rows], y[rows]))
    return rec, offer


def decoupled_actions(models, x, budget):
    rec, offer = models
    rec_scores = np.column_stack([m.predict(x) for m in rec])
    offer_scores = np.column_stack([m.predict(x) for m in offer])
    target = rec_scores.argmax(axis=1) * O + offer_scores.argmax(axis=1)
    rank = rec_scores.max(axis=1) + offer_scores.max(axis=1)
    return apply_budget(target, rank, budget)


def action_features(x, actions):
    one_hot = np.zeros((len(x), R * O))
    one_hot[np.arange(len(x)), actions] = 1
    return np.column_stack((x, one_hot))


def fit_pooled_additive(x, actions, y):
    """Pooled action-feature baseline; intentionally does not include action interactions."""
    return Ridge(alpha=1.0).fit(action_features(x, actions), y)


def pooled_actions(model, x, budget):
    actions = np.arange(R * O)
    predictions = np.column_stack([
        model.predict(action_features(x, np.full(len(x), action))) for action in actions
    ])
    incremental = predictions - predictions[:, [BASELINE_ACTION]]
    return apply_budget(incremental.argmax(axis=1), incremental.max(axis=1), budget)


def apply_budget(target, score, budget):
    treated = np.zeros(len(target), dtype=bool)
    treated[np.argsort(score)[::-1][:max(1, int(budget * len(target)))]] = True
    return np.where(treated, target, BASELINE_ACTION)


def joint_actions(models, x, budget):
    prediction = joint_scores(models, x)
    incremental = prediction - prediction[:, [BASELINE_ACTION]]
    return apply_budget(incremental.argmax(axis=1), incremental.max(axis=1), budget)


def true_incremental_value(hidden_y, chosen_actions):
    flat = hidden_y.reshape(len(hidden_y), R * O)
    return (flat[np.arange(len(hidden_y)), chosen_actions] - flat[:, BASELINE_ACTION]).mean()


def run_replicate(seed, n_train, n_test, lam, exploration, budget):
    x_train, aff_train, ps_train, base_train, coupling = make_population(n_train, seed)
    x_test, aff_test, ps_test, base_test, _ = make_population(n_test, seed + 10_000)
    y_train = potential_outcomes(aff_train, ps_train, base_train, coupling, lam, seed + 20_000)
    y_test = potential_outcomes(aff_test, ps_test, base_test, coupling, lam, seed + 30_000)
    logged_action, propensity = sample_logged_actions(aff_train, ps_train, exploration, seed + 40_000)
    y_observed = observed_outcomes(y_train, logged_action)

    joint = fit_joint_cells(x_train, logged_action, y_observed, seed)
    decoupled = fit_decoupled_marginals(x_train, logged_action, y_observed, seed)
    pooled = fit_pooled_additive(x_train, logged_action, y_observed)

    value_joint = true_incremental_value(y_test, joint_actions(joint, x_test, budget))
    value_decoupled = true_incremental_value(y_test, decoupled_actions(decoupled, x_test, budget))
    value_pooled = true_incremental_value(y_test, pooled_actions(pooled, x_test, budget))
    return {
        "seed": seed,
        "lambda": lam,
        "exploration": exploration,
        "budget": budget,
        "min_logged_propensity": propensity.min(),
        "joint_irc": value_joint,
        "decoupled_irc": value_decoupled,
        "pooled_additive_irc": value_pooled,
        "joint_minus_decoupled": value_joint - value_decoupled,
        "joint_minus_pooled": value_joint - value_pooled,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-test", type=int, default=4000)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.75)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--budget", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=Path("results_logged_policy_learning.csv"))
    args = parser.parse_args()

    rows = [run_replicate(s, args.n_train, args.n_test, args.lam, args.exploration, args.budget)
            for s in range(args.seeds)]
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    summary = result[["joint_irc", "decoupled_irc", "pooled_additive_irc", "joint_minus_decoupled", "joint_minus_pooled"]].agg(["mean", "std"])
    print(summary.round(4))
    print(f"Saved {args.output} with {len(result)} independent simulation replicates.")


if __name__ == "__main__":
    main()
