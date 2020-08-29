"""
This notebook compares the performance of neural networks trained with maximum a
posteriori (MAP) (i.e. maximum likelihood regularized by a prior) and
Hamiltonian Monte Carlo (HMC).
"""


# %%
from functools import partial

import tensorflow_probability as tfp

from bnn import target_log_prob_fn_factory
from bnn.hmc import hmc_predict, predict_from_chain, run_hmc
from bnn.map import map_predict
from data import dropna, load_gaultois, normalize, train_test_split
from utils import predict_multiple_labels

# %%
features, labels = load_gaultois()

labels, features = dropna(labels, features)

features, [X_mean, X_std] = normalize(features)
labels, [y_mean, y_std] = normalize(labels)

[X_train, y_train], [X_test, y_test] = train_test_split(features, labels)

data_sets = [X_train, y_train, X_test, y_test]


# %% [markdown]
# # Maximum A Posteriori Neural Network


# %% Per-label weight and bias priors in the order rho, seebeck, kappa, zT.
weight_priors = [tfp.distributions.Normal(0, std) for std in [0.1, 0.1, 0.1, 0.1]]
bias_priors = [tfp.distributions.Normal(0, std) for std in [0.1, 1.0, 1.0, 1.0]]
map_predictors = [
    partial(map_predict, *priors) for priors in zip(weight_priors, bias_priors)
]


# %%
map_results = predict_multiple_labels(map_predictors, *data_sets)
map_y_preds, map_y_vars, map_log_probs, map_initial_states = map_results


# %% Single-label calculation.
bnn_log_prob_fn = target_log_prob_fn_factory(
    weight_priors[0], bias_priors[0], X_train.values, y_train.rho.values,
)


# %%
burnin, samples, trace, final_kernel_results = run_hmc(
    bnn_log_prob_fn,
    num_results=500,
    num_burnin_steps=1500,
    # current_state=map_initial_states["rho"],
)


# %%
y_pred, y_var = predict_from_chain(samples, X_test.values)


# %%
((y_pred.numpy() - y_test.log_rho_scd.values) ** 2).mean()


# %%
hmc_predictors = [
    partial(hmc_predict, *priors, list(init_state))
    for *priors, init_state in zip(
        weight_priors, bias_priors, map_initial_states.values.T
    )
]


# %%
hmc_results = predict_multiple_labels(hmc_predictors, *data_sets[:3])
hmc_y_preds_scd, hmc_y_vars_scd, *hmc_rest = hmc_results
