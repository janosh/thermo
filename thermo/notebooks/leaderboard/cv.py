"""Cross-validated benchmarks"""


# %%
import pickle
from functools import partial

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from sklearn.model_selection import KFold  # , ShuffleSplit

from thermo.bnn.hmc import hmc_predict
from thermo.bnn.map import map_predict
from thermo.bnn.tf_dropout import do_predict
from thermo.data import dropna, load_gaultois, normalize
from thermo.gp import gp_predict
from thermo.rf import rf_predict
from thermo.utils import ROOT, cross_val_predict, plots
from thermo.utils.evaluate import mae, nxm_to_mxn_cols, plot_output, rmse

# %%
features, targets = load_gaultois()

targets, features = dropna(targets, features)

targets["abs_seebeck"] = abs(targets.seebeck)
log_targets = np.log(targets.drop(columns=["seebeck"]))

X_norm, [X_mean, X_std] = normalize(features)
y_norm, [y_mean, y_std] = normalize(log_targets)

Xy = [X_norm, y_norm]

# Shuffling in KFold is important since the materials in Gaultois'
# database are sorted by family.
kfold = KFold(n_splits=5, shuffle=True, random_state=0)


# %% [markdown]
# # Random Forest Regression


# %%
rf_y_pred, rf_y_var, rf_models = cross_val_predict(kfold, *Xy, rf_predict)


# %%
pd.concat(
    [mae(rf_y_pred, y_norm), rmse(rf_y_pred, y_norm)],
    axis=1,
    keys=["MAE", "RMSE"],
)


# %%
for label, y_test, y_pred, y_std in zip(
    targets.columns, y_norm.values.T, rf_y_pred.values.T, rf_y_var.values.T ** 0.5
):
    plot_output(y_test, y_pred, y_std, title=label)


# %%
rf_out_by_label = nxm_to_mxn_cols(
    [y_norm, rf_y_pred, rf_y_var], keys=["y_test", "y_pred", "y_var"]
)
for df in rf_out_by_label:
    plots.ci_err_decay(df, kfold.n_splits)


# %% [markdown]
# # Maximum A Posteriori Neural Network


# %%
# Per-label weight and bias priors in the order rho, seebeck, kappa, zT.
weight_priors = [tfp.distributions.Normal(0, std) for std in [0.1, 0.1, 0.1, 0.1]]
bias_priors = [tfp.distributions.Normal(0, std) for std in [0.1, 1.0, 1.0, 1.0]]
map_predictors = [
    partial(map_predict, *priors) for priors in zip(weight_priors, bias_priors)
]


# %%
map_y_pred, map_y_var, map_log_probs, map_initial_states = cross_val_predict(
    kfold, *Xy, map_predictors
)


# %%
pd.concat(
    [mae(map_y_pred, y_norm), rmse(map_y_pred, y_norm)],
    axis=1,
    keys=["MAE", "RMSE"],
)


# %%
with open(ROOT + "/results/map_initial_states.pkl", "wb") as file:
    pickle.dump(map_initial_states, file)


# %%
for label, y_test, y_pred, y_std in zip(
    targets.columns,
    y_norm.values.T,
    map_y_pred.values.T,
    map_y_var.values.T ** 0.5,
):
    plot_output(y_test, y_pred, y_std, title=label)


# %%

rf_out_by_label = nxm_to_mxn_cols(
    [y_norm, map_y_pred, map_y_var], keys=["y_test", "y_pred", "y_var"]
)
for df in rf_out_by_label:
    plots.ci_err_decay(df, kfold.n_splits)


# %% [markdown]
# # Dropout Neural Network


# %%
do_predict = partial(do_predict, uncertainty="aleatoric_epistemic", epochs=500)
do_y_pred, do_y_var, do_histories, do_models = cross_val_predict(kfold, *Xy, do_predict)


# %%
pd.concat(
    [mae(do_y_pred, y_norm), rmse(do_y_pred, y_norm)],
    axis=1,
    keys=["MAE", "RMSE"],
)


# %%
for label, y_test, y_pred, y_std in zip(
    targets.columns, y_norm.values.T, do_y_pred.values.T, do_y_var.values.T ** 0.5
):
    plot_output(y_test, y_pred, y_std, title=label)


# %%

rf_out_by_label = nxm_to_mxn_cols(
    [y_norm, do_y_pred, do_y_var], keys=["y_test", "y_pred", "y_var"]
)
for df in rf_out_by_label:
    plots.ci_err_decay(df, kfold.n_splits)


# %% [markdown]
# # Gaussian Process Regression


# %%
gp_y_pred, gp_y_var, gp_models = cross_val_predict(kfold, *Xy, gp_predict)


# %%
pd.concat(
    [mae(gp_y_pred, y_norm), rmse(gp_y_pred, y_norm)],
    axis=1,
    keys=["MAE", "RMSE"],
)


# %%
for label, y_test, y_pred, y_std in zip(
    targets.columns, y_norm.values.T, gp_y_pred.values.T, gp_y_var.values.T ** 0.5
):
    plot_output(y_test, y_pred, y_std, title=label)


# %%
rf_out_by_label = nxm_to_mxn_cols(
    [y_norm, gp_y_pred, gp_y_var], keys=["y_test", "y_pred", "y_var"]
)
for df in rf_out_by_label:
    plots.ci_err_decay(df, kfold.n_splits)


# %% [markdown]
# # Hamiltonian Monte Carlo


# %%
# with open(ROOT + "/results/map_initial_states.pkl", "rb") as file:
#     map_initial_states = pickle.load(file)


# %%
hmc_predictors = [
    partial(hmc_predict, *priors, list(init_state))
    for *priors, init_state in zip(
        weight_priors, bias_priors, map_initial_states[-1].values.T
    )
]


# %%
hmc_y_pred, hmc_y_var, hmc_kernel_results = cross_val_predict(
    kfold, *Xy, hmc_predictors
)


# %%
with open(ROOT + "/results/hmc_results.pkl", "wb") as file:
    pickle.dump([hmc_y_pred, hmc_y_var, hmc_kernel_results], file)


# %%
pd.concat(
    [mae(hmc_y_pred, y_norm), rmse(hmc_y_pred, y_norm)],
    axis=1,
    keys=["MAE", "RMSE"],
)


# %%
for label, y_test, y_pred, y_std in zip(
    targets.columns,
    y_norm.values.T,
    hmc_y_pred.values.T,
    hmc_y_var.values.T ** 0.5,
):
    plot_output(y_test, y_pred, y_std, title=label)


# %%

rf_out_by_label = nxm_to_mxn_cols(
    [y_norm, hmc_y_pred, hmc_y_var], keys=["y_test", "y_pred", "y_var"]
)
for df in rf_out_by_label:
    plots.ci_err_decay(df, kfold.n_splits)


# %% [markdown]
# # Leaderboard


# %%
# all_mses = [rf_mses, map_mses, do_mses, gp_mses, hmc_mses]
# mse_boxes(
#     [df.mse_scd for df in all_mses], ["rf", "map", "do", "gp", "hmc"],
# )


# %%
# Save all numerical results.
all_results = [
    [rf_y_pred, rf_y_var],
    [map_y_pred, map_y_var],
    [do_y_pred, do_y_var],
    [gp_y_pred, gp_y_var],
    [hmc_y_pred, hmc_y_var],
]
with open(ROOT + "/results/leaderboard_cv.pkl", "wb") as file:
    pickle.dump(all_results, file)


# # %%
# # Load all numerical results.
# with open(ROOT + "/results/leaderboard_cv.pkl", "rb") as file:
#     all_results = pickle.load(file)
# [
#     [rf_mses, rf_ys_scd, rf_ys, rf_y_scalers],
#     [map_mses, map_ys_scd, map_ys, map_y_scalers],
#     [do_mses, do_ys_scd, do_ys, do_y_scalers],
#     [gp_mses, gp_ys_scd, gp_ys, gp_y_scalers],
#     [hmc_mses, hmc_ys_scd, hmc_ys, hmc_y_scalers],
# ] = all_results
# rf_y_pred, rf_y_var, rf_y_test = rf_ys
# map_y_pred, map_y_var, map_y_test = map_ys
# do_y_pred, do_y_var, do_y_test = do_ys
# gp_y_pred, gp_y_var, gp_y_test = gp_ys
# hmc_y_pred, hmc_y_var, hmc_y_test = hmc_ys
