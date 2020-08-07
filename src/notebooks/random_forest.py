"""
This notebook evaluates the accuracy and uncertainty estimates of random forest
(RF) trained with Magpie features on predicting electrical resistivity (rho),
Seebeck coefficient (S), thermal conductivity (kappa) and thermoelectric figure
of merit (zT).
"""


# %%
from data import dropna, load_gaultois, normalize, train_test_split
from rf import rf_predict
from utils.evaluate import plot_output, rmse

# %%
features, labels = load_gaultois()

features, [X_mean, X_std] = normalize(features)
labels, [y_mean, y_std] = normalize(labels)

[X_train, y_train], [X_test, y_test] = train_test_split(features, labels)


# %%
rho_train, seebeck_train, kappa_train, zT_train = y_train.to_numpy().T
rho_test, seebeck_test, kappa_test, zT_test = y_test.to_numpy().T


# %% [markdown]
# ## Resistivity model with epistemic uncertainty


# %%
rho_ep_pred, rho_ep_var, rho_ep_model = rf_predict(
    X_train, rho_train, X_test, rho_test, uncertainty="epistemic"
)

# %%
print(f"rho epistemic rmse: {rmse(rho_ep_pred, seebeck_test):.4g}\n")

plot_output(seebeck_test, rho_ep_pred, rho_ep_var ** 0.5, title="rho")


# %% [markdown]
# ## Resistivity model with aleatoric and epistemic uncertainty


# %%
rho_al_ep_pred, rho_al_ep_var, rho_al_ep_model = rf_predict(
    X_train, rho_train, X_test, rho_test
)


# %%
print(f"rho aleatoric_epistemic rmse: {rmse(rho_al_ep_pred, rho_test):.4g}\n")
plot_output(rho_test, rho_al_ep_pred, rho_al_ep_var ** 0.5, title="rho")


# %% [markdown]
# ## Seebeck model with epistemic uncertainty


# %%
see_ep_pred, see_ep_var, see_ep_model = rf_predict(
    X_train, seebeck_train, X_test, seebeck_test, uncertainty="epistemic"
)


# %%
print(f"Seebeck epistemic rmse: {rmse(see_ep_pred, seebeck_test):.4g}\n")
plot_output(seebeck_test, see_ep_pred, see_ep_var ** 0.5, title="seebeck")


# %% [markdown]
# ## Seebeck model with aleatoric uncertainty


# %%
see_al_pred, see_al_var, see_al_model = rf_predict(
    X_train,
    seebeck_train,
    X_test,
    seebeck_test,
    uncertainty="aleatoric",
    min_samples_leaf=10,
)


# %%
print(f"Seebeck aleatoric rmse: {rmse(see_al_pred, seebeck_test):.4g}\n")
plot_output(seebeck_test, see_al_pred, see_al_var ** 0.5, title="seebeck")


# %% [markdown]
# ## Seebeck model with aleatoric and epistemic uncertainty


# %%
see_al_ep_pred, see_al_ep_var, see_al_ep_model = rf_predict(
    X_train, seebeck_train, X_test, seebeck_test
)


# %%
print(f"seebeck aleatoric_epistemic rmse: {rmse(see_al_ep_pred, seebeck_test):.4g}\n")
plot_output(seebeck_test, see_al_ep_pred, see_al_ep_var ** 0.5, title="seebeck")


# %% [markdown]
# # Remove NaNs from kappa and zT


# %%
kappa_train, zT_train, X_train_no_nan = dropna(kappa_train, zT_train, X_train)
kappa_test, zT_test, X_test_no_nan = dropna(kappa_test, zT_test, X_test)


# %% [markdown]
# ## Thermal conductivity model with epistemic uncertainty


# %%
kappa_ep_pred, kappa_ep_var, kappa_ep_model = rf_predict(
    X_train_no_nan, kappa_train, X_test_no_nan, kappa_test, uncertainty="epistemic"
)


# %%
print(f"kappa epistemic rmse: {rmse(kappa_ep_pred, kappa_test):.4g}\n")
plot_output(kappa_test, kappa_ep_pred, kappa_ep_var ** 0.5, title="kappa")


# %% [markdown]
# ## Thermal conductivity model with aleatoric uncertainty
# Aleatoric uncertainty alone doesn't work well (unlike epistemic variance derived
# from the ensemble of trees).


# %%
kappa_al_pred, kappa_al_var, kappa_al_model = rf_predict(
    X_train_no_nan,
    kappa_train,
    X_test_no_nan,
    kappa_test,
    uncertainty="aleatoric",
    min_samples_leaf=10,
)


# %%
print(f"kappa aleatoric rmse: {rmse(kappa_al_pred, kappa_test):.4g}\n")
plot_output(kappa_test, kappa_al_pred, kappa_al_var ** 0.5, title="kappa")


# %% [markdown]
# ## Thermal conductivity model with aleatoric and epistemic uncertainty


# %%
kappa_al_ep_pred, kappa_al_ep_var, _ = rf_predict(
    X_train_no_nan, kappa_train, X_test_no_nan, kappa_test
)


# %%
print(f"kappa aleatoric_epistemic rmse: {rmse(kappa_al_ep_pred, kappa_test):.4g}\n")
plot_output(kappa_test, kappa_al_ep_pred, kappa_al_ep_var ** 0.5, title="kappa")


# %% [markdown]
# ## zT model with epistemic uncertainty


# %%
zT_ep_pred, zT_ep_var, zT_ep_model = rf_predict(
    X_train_no_nan, zT_train, X_test_no_nan, zT_test, uncertainty="epistemic"
)


# %%
print(f"zT epistemic rmse: {rmse(zT_ep_pred, zT_test):.4g}\n")
plot_output(zT_test, zT_ep_pred, zT_ep_var ** 0.5, title="zT")


# %% [markdown]
# ## zT model with aleatoric uncertainty


# %%
zT_al_pred, zT_al_var, zT_al_model = rf_predict(
    X_train_no_nan, zT_train, X_test_no_nan, zT_test, uncertainty="aleatoric",
)


# %%
print(f"zT aleatoric rmse: {rmse(zT_al_pred, zT_test):.4g}\n")
plot_output(zT_test, zT_al_pred, zT_al_var ** 0.5, title="zT")


# %% [markdown]
# ## zT model with aleatoric and epistemic uncertainty


# %%
zT_al_ep_pred, zT_al_ep_var, _ = rf_predict(
    X_train_no_nan, zT_train, X_test_no_nan, zT_test
)


# %%
print(f"zT aleatoric_epistemic rmse: {rmse(zT_al_ep_pred, zT_test):.4g}\n")
plot_output(zT_test, zT_al_ep_pred, zT_al_ep_var ** 0.5, title="zT")
