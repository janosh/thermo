# %%
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from thermo.bnn.tf_dropout import do_predict
from thermo.data import dropna, load_gaultois, normalize
from thermo.evaluate import df_corr
from thermo.plots import plot_output
from thermo.utils import ROOT, cross_val_predict


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
# # Aleatoric Uncertainty


# %%
al_predict = partial(do_predict, uncertainty="aleatoric", epochs=1000, n_preds=1000)

al_y_pred, al_y_var, al_histories, al_models = cross_val_predict(kfold, *Xy, al_predict)


# %%
pd.concat(
    [abs(al_y_pred - y_norm).mean(), ((al_y_pred - y_norm) ** 2).mean()],
    axis=1,
    keys=["MAE", "MSE"],
)


# %%
for label, y_test, y_pred, y_var in zip(
    targets.columns, y_norm.values.T, al_y_pred.values.T, al_y_var.values.T
):
    plot_output(y_test, y_pred, y_var**0.5, title=label)


# %%
print("correlation between aleatoric absolute error and standard deviation")
al_err_std_corr = df_corr(abs(y_norm - al_y_pred), al_y_var)
al_err_std_corr


# %% [markdown]
# # Epistemic Uncertainty


# %%
ep_predict = partial(do_predict, uncertainty="epistemic", epochs=1000, n_preds=1000)

ep_y_pred, ep_y_var, ep_histories, ep_models = cross_val_predict(kfold, *Xy, ep_predict)


# %%
pd.concat(
    [abs(ep_y_pred - y_norm).mean(), ((ep_y_pred - y_norm) ** 2).mean()],
    axis=1,
    keys=["MAE", "MSE"],
)


# %%
for label, y_test, y_pred, y_var in zip(
    targets.columns, y_norm.values.T, ep_y_pred.values.T, ep_y_var.values.T
):
    plot_output(y_test, y_pred, y_var**0.5, title=label)


# %%
print("correlation between epistemic absolute error and standard deviation")
ep_err_std_corr = df_corr(abs(y_norm - ep_y_pred), ep_y_var)
ep_err_std_corr


# %% [markdown]
# # Aleatoric + Epistemic Uncertainty


# %%
al_ep_predict = partial(
    do_predict, uncertainty="aleatoric_epistemic", epochs=1000, n_preds=1000
)

al_ep_y_pred, al_ep_y_var, al_ep_histories, al_ep_models = cross_val_predict(
    kfold, *Xy, al_ep_predict
)


# %%
pd.concat(
    [abs(al_ep_y_pred - y_norm).mean(), ((al_ep_y_pred - y_norm) ** 2).mean()],
    axis=1,
    keys=["MAE", "MSE"],
)


# %%
for label, y_test, y_pred, y_var in zip(
    targets.columns, y_norm.values.T, al_ep_y_pred.values.T, al_ep_y_var.values.T
):
    plot_output(y_test, y_pred, y_var**0.5, title=label)


# %%
print("correlation between aleatoric+epistemic absolute error and standard deviation")
al_ep_err_std_corr = df_corr(abs(y_norm - al_ep_y_pred), al_ep_y_var)
al_ep_err_std_corr


# %%
pd.concat(
    [al_err_std_corr, ep_err_std_corr, al_ep_err_std_corr],
    axis=1,
    keys=["aleatoric", "epistemic", "aleatoric + epistemic"],
).to_latex(
    ROOT + "/results/al_vs_ep/do/abs_err_vs_y_std_corr.tex",
    escape=False,
    float_format="%.3g",
    multicolumn_format="c",
)


# %% [markdown]
# # Epistemic-Aleatoric Correlation


# %%
print("Aleatoric vs epistemic correlation:")
al_ep_corr = df_corr(al_y_var, ep_y_var)
al_ep_corr


# %%
print("Aleatoric vs aleatoric_epistemic correlation:")
al_al_ep_corr = df_corr(al_y_var, al_ep_y_var)
al_al_ep_corr


# %%
print("Epistemic vs aleatoric_epistemic correlation:")
ep_al_ep_corr = df_corr(ep_y_var, al_ep_y_var)
ep_al_ep_corr


# %%
pd.concat(
    [al_ep_corr, al_al_ep_corr, ep_al_ep_corr],
    axis=1,
    keys=["aleatoric vs epistemic", "aleatoric vs full", "epistemic vs full"],
).to_latex(
    ROOT + "/results/al_vs_ep/do/al_ep_corr.tex",
    escape=False,
    float_format="%.3g",
    multicolumn_format="c",
)
