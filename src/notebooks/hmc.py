"""
This notebook compares the performance of neural networks trained with maximum a
posteriori (MAP) (i.e. maximum likelihood regularized by a prior) and
Hamiltonian Monte Carlo (HMC).
"""


# %%
from functools import partial

import tensorflow_probability as tfp

from bnn.hmc import hmc_predict
from bnn.map import map_predict
from data import load_gaultois, normalize, train_test_split
from utils import predict_multiple_labels
from utils.evaluate import get_mses_as_df

# %%
features, labels = load_gaultois()

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
map_y_preds_scd, map_y_vars_scd, map_log_probs, map_initial_states = map_results


# %%
map_mses = get_mses_as_df([y_test, map_y_preds_scd])
map_mses


# # %% Single-label calculation.
# bnn_log_prob_fn = bnn_fn.target_log_prob_fn_factory(
#     weight_priors[0],
#     bias_priors[0],
#     X_train.values.astype("float32"),
#     y_train.log_rho_scd.values.astype("float32"),
# )


# # %%
# burnin, samples, trace, final_kernel_results = run_hmc(
#     bnn_log_prob_fn,
#     num_results=500,
#     num_burnin_steps=1500,
#     current_state=list(map_initial_states.log_rho_scd),
# )


# # %%
# y_pred, y_var = predict_from_chain(samples, X_test.values.astype("float32"))


# # %%
# ((y_pred.numpy() - y_test.log_rho_scd.values) ** 2).mean()


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


# %%
hmc_mses = get_mses_as_df([y_test, hmc_y_preds_scd])
hmc_mses
