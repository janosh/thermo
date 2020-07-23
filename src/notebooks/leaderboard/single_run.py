# %%
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import Markdown, display

from bnn.map import map_predict
from bnn.tf_dropout import do_predict, save_loss_history
from data import dropna, load_gaultois, normalize, train_test_split
from gp import gp_predict
from rf import rf_predict
from utils import ROOT, plots, predict_multiple_labels
from utils.evaluate import (
    back_transform_labels,
    compute_zT,
    mae,
    plot_output,
    rmse,
)

# %%
features, labels = load_gaultois()

labels, features = dropna(labels, features)

labels["abs_seebeck"] = abs(labels.seebeck)
log_labels = np.log(labels.drop(columns=["seebeck"]))

norm_features, [X_mean, X_std] = normalize(features)
norm_labels, [y_mean, y_std] = normalize(log_labels)

[X_train, y_train], [X_test, y_test] = train_test_split(norm_features, norm_labels)

Xy = [X_train, y_train.T, X_test, y_test.T]


# %% [markdown]
# # Random Forest Regression


# %%
rf_y_pred, rf_y_var, rf_models = predict_multiple_labels(rf_predict, *Xy)


# %%
rf_y_pred_orig, rf_y_var_orig = back_transform_labels(
    y_mean, y_std, rf_y_pred, rf_y_var, to="orig"
)


# %%
rf_mae = mae(y_test, rf_y_pred).rename("MAE")
rf_rmse = rmse(y_test, rf_y_pred).rename("RMSE")
rf_mae.to_frame().join(rf_rmse)


# %%
for label, y_true, y_pred, y_var in zip(
    y_test.columns, y_test.values.T, rf_y_pred.values.T, rf_y_var.values.T,
):
    display(Markdown(f"# {label}"))
    plot_output(y_true, y_pred, y_var ** 0.5, title=label)


# %%
rf_y_pred_orig["zT_computed"] = compute_zT(
    rf_y_pred_orig.rho,
    rf_y_pred_orig.abs_seebeck,
    rf_y_pred_orig.kappa,
    features["T"].loc[y_test.index],
)

zT_rmse = rmse(labels.zT.loc[y_test.index], rf_y_pred_orig.zT_computed)

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
map_results = predict_multiple_labels(map_predictors, *Xy)
map_y_pred, map_y_var, map_log_probs, map_initial_states = map_results


# %%
map_y_pred_orig, map_y_var_orig = back_transform_labels(
    y_mean, y_std, map_y_pred, map_y_var, to="orig"
)


# %%
map_mae = mae(y_test, map_y_pred).rename("MAE")
map_rmse = rmse(y_test, map_y_pred).rename("RMSE")
map_mae.to_frame().join(map_rmse)


# %%
for label, map_log_prob in zip(labels.columns, map_log_probs.values.T):
    display(Markdown(f"# {label}"))
    plots.log_probs(zip(["train", "test"], map_log_prob), title=label)


# %%
for label, y_true, y_pred, y_var in zip(
    labels.columns, y_test.values.T, map_y_pred.values.T, map_y_var.values.T,
):
    display(Markdown(f"# {label}"))
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

gp_y_pred, gp_y_var, gp_models = predict_multiple_labels(gp_predict, *Xy)


# %%
gp_y_pred_orig, gp_y_var_orig = back_transform_labels(
    y_mean, y_std, gp_y_pred, gp_y_var, to="orig"
)


# %%
gp_mae = mae(y_test, gp_y_pred).rename("MAE")
gp_rmse = rmse(y_test, gp_y_pred).rename("RMSE")
gp_mae.to_frame().join(gp_rmse)


# %%
for label, y_true, y_pred, y_std in zip(
    labels.columns, y_test.values.T, gp_y_pred.values.T, gp_y_var.values.T ** 0.5,
):
    display(Markdown(f"# {label}"))
    plot_output(y_true, y_pred, y_std, title=label)


# %% [markdown]
# # Dropout Neural Network


# %%
# one of ["aleatoric", "epistemic", "aleatoric_epistemic"]
do_uncertainty = "aleatoric_epistemic"
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=ROOT + "/logs/leaderboard/single-runs")

do_y_pred, do_y_var, do_histories, do_models = predict_multiple_labels(
    partial(do_predict, uncertainty=do_uncertainty, cbs=[tb_cb], epochs=500), *Xy,
)


# %%
do_y_pred_orig, do_y_var_orig = back_transform_labels(
    y_mean, y_std, do_y_pred, do_y_var, to="orig"
)


# %%
do_mae = mae(y_test, do_y_pred).rename("MAE")
do_rmse = rmse(y_test, do_y_pred).rename("RMSE")
do_mae.to_frame().join(do_rmse)


# %%
for label, y_true, y_pred, y_std, hist in zip(
    labels.columns,
    y_test.values.T,
    do_y_pred.values.T,
    do_y_var.values.T ** 0.5,
    do_histories.values(),
    # models,
):
    display(Markdown(f"# {label}"))
    save_loss_history(hist)
    plot_output(y_true, y_pred, y_std, title=label)
    # plot_model(model, log_dir)
