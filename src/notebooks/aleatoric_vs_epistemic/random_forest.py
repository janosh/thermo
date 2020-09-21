# %%
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from data import dropna, load_gaultois, normalize
from rf import rf_predict
from utils import ROOT, cross_val_predict
from utils.evaluate import df_corr, mae, plot_output, rmse

# %%
features, labels = load_gaultois()

labels, features = dropna(labels, features)

labels["abs_seebeck"] = abs(labels.seebeck)
log_labels = np.log(labels.drop(columns=["seebeck"]))

X_norm, [X_mean, X_std] = normalize(features)
y_norm, [y_mean, y_std] = normalize(log_labels)

Xy = [X_norm, y_norm]

# Shuffling in KFold is important since the materials in Gaultois'
# database are sorted by family.
kfold = KFold(n_splits=5, shuffle=True, random_state=0)


# %% [markdown]
# # Aleatoric Uncertainty


# %%
rf_al_predict = partial(rf_predict, min_samples_leaf=10, uncertainty="aleatoric")

al_y_pred, al_y_var, al_models = cross_val_predict(kfold, *Xy, rf_al_predict)


# %%
pd.concat(
    [mae(al_y_pred, y_norm), rmse(al_y_pred, y_norm)], axis=1, keys=["MAE", "RMSE"],
)


# %%
for label, y_test, y_pred, y_var in zip(
    labels.columns, y_norm.values.T, al_y_pred.values.T, al_y_var.values.T
):
    plot_output(y_test, y_pred, y_var ** 0.5, title=label)


# %%
print("correlation between aleatoric absolute error and standard deviation")
al_err_std_corr = df_corr(abs(y_norm - al_y_pred), al_y_var ** 0.5)
al_err_std_corr


# %% [markdown]
# # Epistemic Uncertainty


# %%
ep_y_pred, ep_y_var, ep_models = cross_val_predict(kfold, *Xy, rf_predict)


# %%
pd.concat(
    [mae(ep_y_pred, y_norm), rmse(ep_y_pred, y_norm)], axis=1, keys=["MAE", "RMSE"],
)


# %%
for label, y_test, y_pred, y_var in zip(
    labels.columns, y_norm.values.T, ep_y_pred.values.T, ep_y_var.values.T
):
    plot_output(y_test, y_pred, y_var ** 0.5, title=label)


# %%
print("correlation between epistemic absolute error and standard deviation")
ep_err_std_corr = df_corr(abs(y_norm - ep_y_pred), ep_y_var ** 0.5)
ep_err_std_corr


# %% [markdown]
# # Aleatoric + Epistemic Uncertainty


# %%
rf_al_ep_predict = partial(rf_predict, min_samples_leaf=10)

al_ep_y_pred, al_ep_y_var, al_ep_models = cross_val_predict(
    kfold, *Xy, rf_al_ep_predict
)


# %%
pd.concat(
    [mae(al_ep_y_pred, y_norm), rmse(al_ep_y_pred, y_norm)],
    axis=1,
    keys=["MAE", "RMSE"],
)


# %%
for label, y_test, y_pred, y_var in zip(
    labels.columns, y_norm.values.T, al_ep_y_pred.values.T, al_ep_y_var.values.T
):
    plot_output(y_test, y_pred, y_var ** 0.5, title=label)


# %%
print("correlation between aleatoric+epistemic absolute error and standard deviation")
al_ep_err_std_corr = df_corr(abs(y_norm - al_ep_y_pred), al_ep_y_var ** 0.5)
al_ep_err_std_corr


# %%
pd.concat(
    [al_err_std_corr, ep_err_std_corr, al_ep_err_std_corr],
    axis=1,
    keys=["aleatoric", "epistemic", "aleatoric + epistemic"],
).to_latex(
    ROOT + "/results/al_vs_ep/rf/abs_err_vs_y_std_corr.tex",
    escape=False,
    float_format="%.3g",
    multicolumn_format="c",
)


# %% [markdown]
# # Epistemic-Aleatoric Correlation


# %%
print("Aleatoric vs epistemic correlation")
al_ep_corr = df_corr(al_y_var, ep_y_var)
al_ep_corr


# %%
print("Aleatoric vs aleatoric_epistemic correlation")
al_al_ep_corr = df_corr(al_y_var, al_ep_y_var)
al_al_ep_corr


# %%
print("Epistemic vs aleatoric_epistemic correlation")
ep_al_ep_corr = df_corr(ep_y_var, al_ep_y_var)
ep_al_ep_corr


# %%
pd.concat(
    [al_ep_corr, al_al_ep_corr, ep_al_ep_corr],
    axis=1,
    keys=["aleatoric vs epistemic", "aleatoric vs full", "epistemic vs full"],
).to_latex(
    ROOT + "/results/al_vs_ep/rf/al_ep_corr.tex",
    escape=False,
    float_format="%.3g",
    multicolumn_format="c",
)
