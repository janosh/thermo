"""This notebook evaluates the accuracy and uncertainty estimates of dropout neural
networks (DNN) trained with Magpie features on predicting electrical resistivity
(rho), Seebeck coefficient (S), thermal conductivity (kappa) and thermoelectric
figure of merit (zT).
"""

# %%
import matplotlib.pyplot as plt

from thermo.bnn.tf_dropout import do_predict
from thermo.data import load_gaultois, normalize, train_test_split
from thermo.evaluate import rmse
from thermo.plots import plot_output


# %%
features, targets = load_gaultois()

features, [X_mean, X_std] = normalize(features)
targets, [y_mean, y_std] = normalize(targets)

[X_train, y_train], [X_test, y_test] = train_test_split(features, targets)


# %%
def plot_loss_history(hist):
    fig = plt.figure(figsize=[12, 5])
    for key, data in hist.items():
        if "loss" in key:
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(data, label=key)
            ax1.set(xlabel="epoch")
        else:  # plot other metrics like accuracy or loss without
            # regularizers on a separate axis
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(data, label=key)
            ax2.set(xlabel="epoch")

    [ax.legend() for ax in fig.axes]

    plt.show()


# %%
rho_train, seebeck_train, kappa_train, zT_train = y_train.to_numpy().T
rho_test, seebeck_test, kappa_test, zT_test = y_test.to_numpy().T


# %% [markdown]
# # Resistivity model with epistemic uncertainty


# %%
rho_ep_pred, rho_ep_var, rho_ep_history, rho_ep_model = do_predict(
    X_train, rho_train, X_test, rho_test, uncertainty="epistemic"
)


# %%
print(f"rho epistemic rmse: {rmse(rho_ep_pred, rho_test)}")

plot_loss_history(rho_ep_history)

plot_output(rho_test, rho_ep_pred, rho_ep_var**0.5, title="rho")


# %% [markdown]
# # Resistivity model with aleatoric and epistemic uncertainty


# %%
rho_al_ep_pred, rho_al_ep_var, rho_al_ep_history, rho_al_ep_model = do_predict(
    X_train, rho_train, X_test, rho_test
)


# %%
print(f"rho aleatoric_epistemic rmse: {rmse(rho_al_ep_pred, rho_test)}")
plot_loss_history(rho_al_ep_history)
plot_output(rho_test, rho_al_ep_pred, rho_al_ep_var**0.5, title="rho")


# %% [markdown]
# # Seebeck model with epistemic uncertainty


# %%
see_ep_pred, see_ep_var, see_ep_history, see_ep_model = do_predict(
    X_train, seebeck_train, X_test, seebeck_test, uncertainty="epistemic"
)


# %%
print(f"Seebeck epistemic rmse: {rmse(see_ep_pred, seebeck_test)}")
plot_loss_history(see_ep_history)
plot_output(seebeck_test, see_ep_pred, see_ep_var**0.5, title="seebeck")


# %% [markdown]
# # Seebeck model with aleatoric and epistemic uncertainty


# %%
see_al_ep_pred, see_al_ep_var, see_al_ep_history, see_al_ep_model = do_predict(
    X_train, seebeck_train, X_test, seebeck_test
)


# %%
print(f"seebeck aleatoric_epistemic rmse: {rmse(see_al_ep_pred, seebeck_test)}")
plot_loss_history(see_al_ep_history)
plot_output(seebeck_test, see_al_ep_pred, see_al_ep_var**0.5, title="seebeck")


# %% [markdown]
# # Thermal conductivity model with epistemic uncertainty


# %%
kappa_ep_pred, kappa_ep_var, kappa_ep_history, kappa_ep_model = do_predict(
    X_train, kappa_train, X_test, kappa_test, uncertainty="epistemic"
)


# %%
print(f"kappa epistemic rmse: {rmse(kappa_ep_pred, kappa_test)}")
plot_loss_history(kappa_ep_history)
plot_output(kappa_test, kappa_ep_pred, kappa_ep_var**0.5, title="kappa")


# %% [markdown]
# # Thermal conductivity model with aleatoric and epistemic uncertainty


# %%
kappa_al_ep_pred, kappa_al_ep_var, kappa_al_ep_history, _ = do_predict(
    X_train, kappa_train, X_test, kappa_test
)


# %%
print(f"kappa aleatoric_epistemic rmse: {rmse(kappa_al_ep_pred, kappa_test)}")
plot_loss_history(kappa_al_ep_history)
plot_output(kappa_test, kappa_al_ep_pred, kappa_al_ep_var**0.5, title="kappa")
