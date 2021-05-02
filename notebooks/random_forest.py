"""
This notebook evaluates the accuracy and uncertainty estimates of random forest
(RF) trained with Magpie features on predicting electrical resistivity (rho),
Seebeck coefficient (S), thermal conductivity (kappa) and thermoelectric figure
of merit (zT).
"""


# %%
import numpy as np

from thermo.data import dropna, load_gaultois, train_test_split
from thermo.plots import plot_output
from thermo.rf import rf_predict


# %%
features, targets = load_gaultois()

targets["abs_seebeck"] = abs(targets.seebeck)
# see unlog_preds() for how to compute original variance of log-normal RV
log_targets = np.log(targets.drop(columns=["seebeck"]))

[X_train, y_train], [X_test, y_test] = train_test_split(features, log_targets)


# %%
rho_train, kappa_train, zT_train, seebeck_train = y_train.to_numpy().T
rho_test, kappa_test, zT_test, seebeck_test = y_test.to_numpy().T


# %%
# Remove NaNs from kappa and zT
kappa_train, zT_train, X_train_no_nan = dropna(kappa_train, zT_train, X_train)
kappa_test, zT_test, X_test_no_nan = dropna(kappa_test, zT_test, X_test)


# %% [markdown]
# ## Resistivity model with epistemic uncertainty


# %%
rho_ep_pred, rho_ep_std, _ = rf_predict(
    X_train, rho_train, X_test, uncertainty="epistemic"
)


# %%
print(f"rho epistemic MAE: {abs(rho_ep_pred - rho_test).mean():.3f}\n")

plot_output(rho_test, rho_ep_pred, rho_ep_std, title="rho")


# %% [markdown]
# ## Resistivity model with aleatoric and epistemic uncertainty


# %%
rho_al_ep_pred, rho_al_ep_std, _ = rf_predict(X_train, rho_train, X_test)


# %%
print(f"rho aleatoric_epistemic MAE: {abs(rho_al_ep_pred - rho_test).mean():.3f}\n")
plot_output(rho_test, rho_al_ep_pred, rho_al_ep_std, title="rho")


# %% [markdown]
# ## Seebeck model with epistemic uncertainty


# %%
see_ep_pred, see_ep_std, _ = rf_predict(
    X_train, seebeck_train, X_test, uncertainty="epistemic"
)


# %%
print(f"Seebeck epistemic MAE: {abs(see_ep_pred - seebeck_test).mean():.3f}\n")
plot_output(seebeck_test, see_ep_pred, see_ep_std, title="seebeck")


# %% [markdown]
# ## Seebeck model with aleatoric uncertainty


# %%
see_al_pred, see_al_std, _ = rf_predict(
    X_train,
    seebeck_train,
    X_test,
    uncertainty="aleatoric",
    min_samples_leaf=10,
)


# %%
print(f"Seebeck aleatoric MAE: {abs(see_al_pred - seebeck_test).mean():.3f}\n")
plot_output(seebeck_test, see_al_pred, see_al_std, title="seebeck")


# %% [markdown]
# ## Seebeck model with aleatoric and epistemic uncertainty


# %%
see_al_ep_pred, see_al_ep_std, _ = rf_predict(X_train, seebeck_train, X_test)


# %%
print(
    "seebeck aleatoric_epistemic MAE:"
    f" {abs(see_al_ep_pred - seebeck_test).mean():.3f}\n"
)
plot_output(seebeck_test, see_al_ep_pred, see_al_ep_std, title="seebeck")


# %% [markdown]
# ## Thermal conductivity model with epistemic uncertainty


# %%
kappa_ep_pred, kappa_ep_std, _ = rf_predict(
    X_train_no_nan, kappa_train, X_test_no_nan, uncertainty="epistemic"
)


# %%
print(f"kappa epistemic MAE: {abs(kappa_ep_pred - kappa_test).mean():.3f}\n")
plot_output(kappa_test, kappa_ep_pred, kappa_ep_std, title="kappa")


# %% [markdown]
# ## Thermal conductivity model with aleatoric uncertainty
# Aleatoric uncertainty alone doesn't work well (unlike epistemic variance derived
# from the ensemble of trees).


# %%
kappa_al_pred, kappa_al_std, _ = rf_predict(
    X_train_no_nan,
    kappa_train,
    X_test_no_nan,
    uncertainty="aleatoric",
    min_samples_leaf=10,
)


# %%
print(f"kappa aleatoric MAE: {abs(kappa_al_pred - kappa_test).mean():.3f}\n")
plot_output(kappa_test, kappa_al_pred, kappa_al_std, title="kappa")


# %% [markdown]
# ## Thermal conductivity model with aleatoric and epistemic uncertainty


# %%
kappa_al_ep_pred, kappa_al_ep_std, _ = rf_predict(
    X_train_no_nan, kappa_train, X_test_no_nan
)


# %%
print(
    f"kappa aleatoric_epistemic MAE: {abs(kappa_al_ep_pred - kappa_test).mean():.3f}\n"
)
plot_output(kappa_test, kappa_al_ep_pred, kappa_al_ep_std, title="kappa")


# %% [markdown]
# ## zT model with epistemic uncertainty


# %%
zT_ep_pred, zT_ep_std, _ = rf_predict(
    X_train_no_nan, zT_train, X_test_no_nan, uncertainty="epistemic"
)


# %%
print(f"zT epistemic MAE: {abs(zT_ep_pred - zT_test).mean():.3f}\n")
plot_output(zT_test, zT_ep_pred, zT_ep_std, title="zT")


# %% [markdown]
# ## zT model with aleatoric uncertainty


# %%
zT_al_pred, zT_al_std, _ = rf_predict(
    X_train_no_nan, zT_train, X_test_no_nan, uncertainty="aleatoric"
)


# %%
print(f"zT aleatoric MAE: {abs(zT_al_pred - zT_test).mean():.3f}\n")
plot_output(zT_test, zT_al_pred, zT_al_std, title="zT")


# %% [markdown]
# ## zT model with aleatoric and epistemic uncertainty


# %%
zT_al_ep_pred, zT_al_ep_std, _ = rf_predict(X_train_no_nan, zT_train, X_test_no_nan)


# %%
print(f"zT aleatoric_epistemic MAE: {abs(zT_al_ep_pred - zT_test).mean():.3f}\n")
plot_output(zT_test, zT_al_ep_pred, zT_al_ep_std, title="zT")
