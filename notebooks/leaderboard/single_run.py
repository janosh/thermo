# %%
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from thermo.bnn.map import map_predict
from thermo.bnn.tf_dropout import do_predict
from thermo.data import dropna, load_gaultois, normalize, train_test_split
from thermo.evaluate import compute_zT, denorm, mae, rmse
from thermo.gp import gp_predict
from thermo.plots import plot_output
from thermo.rf import rf_predict
from thermo.utils import ROOT, predict_multiple_targets


# %%
features, targets = load_gaultois()

targets, features = dropna(targets, features)

targets["abs_seebeck"] = abs(targets.seebeck)
log_targets = np.log(targets.drop(columns=["seebeck"]))

norm_features, [X_mean, X_std] = normalize(features)
norm_targets, [y_mean, y_std] = normalize(log_targets)

[X_train, y_train], [X_test, y_test] = train_test_split(norm_features, norm_targets)

Xy = [X_train, y_train, X_test, y_test]


# %%
def plot_log_probs(log_probs: list, title: str = None) -> None:
    for legend_label, log_prob in log_probs:
        plt.plot(log_prob, label=legend_label)

    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.title(title)


# %% [markdown]
# # Random Forest Regression


# %%
rf_y_pred, rf_y_var, rf_models = predict_multiple_targets(rf_predict, *Xy)


# %%
rf_y_pred_orig, rf_y_var_orig = denorm(y_mean, y_std, rf_y_pred, rf_y_var)


# %%
rf_mae = mae(y_test, rf_y_pred).rename("MAE")
rf_rmse = rmse(y_test, rf_y_pred).rename("RMSE")
rf_mae.to_frame().join(rf_rmse)


# %%
for label, y_true, y_pred, y_var in zip(
    y_test.columns, y_test.values.T, rf_y_pred.values.T, rf_y_var.values.T
):
    print(f"\n{label}\n")
    plot_output(y_true, y_pred, y_var ** 0.5, title=label)


# %%
rf_y_pred_orig["zT_computed"] = compute_zT(
    rf_y_pred_orig.rho,
    rf_y_pred_orig.abs_seebeck,
    rf_y_pred_orig.kappa,
    features["T"].loc[y_test.index],
)

zT_rmse = rmse(targets.zT.loc[y_test.index], rf_y_pred_orig.zT_computed)

print(f"zT_log_computed mse: {zT_rmse:.4g}")


# %% [markdown]
# # Maximum A Posteriori Neural Network


# %%
# Per-label weight and bias priors in the order rho, seebeck, kappa, zT.
weight_priors = [tfp.distributions.Normal(0, std) for std in [0.1, 0.1, 0.1, 0.1]]
bias_priors = [tfp.distributions.Normal(0, std) for std in [1.0, 1.0, 1.0, 1.0]]
map_predictors = [
    partial(map_predict, *priors) for priors in zip(weight_priors, bias_priors)
]


# %%
map_results = predict_multiple_targets(map_predictors, *Xy)
map_y_pred, map_y_var, map_log_probs, map_initial_states = map_results


# %%
map_y_pred_orig, map_y_var_orig = denorm(y_mean, y_std, map_y_pred, map_y_var)


# %%
map_mae = mae(y_test, map_y_pred).rename("MAE")
map_rmse = rmse(y_test, map_y_pred).rename("RMSE")
map_mae.to_frame().join(map_rmse)


# %%
for label, map_log_prob in zip(targets.columns, map_log_probs.values.T):
    print(f"\n{label}\n")
    plot_log_probs(zip(["train", "test"], map_log_prob), title=label)
    plt.show()


# %%
for label, y_true, y_pred, y_var in zip(
    targets.columns,
    y_test.values.T,
    map_y_pred.values.T,
    map_y_var.values.T,
):
    print(f"\n{label}\n")
    plot_output(y_true, y_pred, y_var ** 0.5, title=label)


# %%
map_y_pred_orig["zT_log_computed"] = compute_zT(
    *map_y_pred_orig.values[:3].T, y_test["T_log"]
).reset_index(drop=True)

zT_rmse = rmse(y_test["zT_log"], map_y_pred_orig["zT_log_computed"])
print(f"zT_log_computed mse: {zT_rmse:.4g}")


# %% [markdown]
# # Gaussian Process Regression


# %%
# GPR might benefit from PCA
# from sklearn.decomposition import PCA
# n_features = X_train.shape[1]
# pca = PCA(n_components=n_features // 4)
# X_train = pca.fit(X_train).transform(X_train)
# X_test = pca.transform(X_test)

gp_y_pred, gp_y_var, gp_models = predict_multiple_targets(gp_predict, *Xy)


# %%
gp_y_pred_orig, gp_y_var_orig = denorm(y_mean, y_std, gp_y_pred, gp_y_var)


# %%
gp_mae = mae(y_test, gp_y_pred).rename("MAE")
gp_rmse = rmse(y_test, gp_y_pred).rename("RMSE")
gp_mae.to_frame().join(gp_rmse)


# %%
for label, y_true, y_pred, y_std in zip(
    targets.columns, y_test.values.T, gp_y_pred.values.T, gp_y_var.values.T ** 0.5
):
    print(f"\n{label}\n")
    plot_output(y_true, y_pred, y_std, title=label)


# %% [markdown]
# # Dropout Neural Network


# %%
# one of ["aleatoric", "epistemic", "aleatoric_epistemic"]
do_uncertainty = "aleatoric_epistemic"
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=ROOT + "/logs/leaderboard/single-runs")

do_y_pred, do_y_var, do_histories, do_models = predict_multiple_targets(
    partial(do_predict, uncertainty=do_uncertainty, cbs=[tb_cb], epochs=500), *Xy
)


# %%
do_y_pred_orig, do_y_var_orig = denorm(y_mean, y_std, do_y_pred, do_y_var)


# %%
do_mae = mae(y_test, do_y_pred).rename("MAE")
do_rmse = rmse(y_test, do_y_pred).rename("RMSE")
do_mae.to_frame().join(do_rmse)


# %%
for label, y_true, y_pred, y_std, hist in zip(
    targets.columns,
    y_test.values.T,
    do_y_pred.values.T,
    do_y_var.values.T ** 0.5,
    do_histories.values(),
    # models,
):
    print(f"\n{label}\n")
    plot_output(y_true, y_pred, y_std, title=label)
