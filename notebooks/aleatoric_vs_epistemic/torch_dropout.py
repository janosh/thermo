# %%
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from thermo.bnn.torch_dropout import GaultoisData, TorchDropoutModel
from thermo.plots import plot_output


# %%
train_set = GaultoisData(target_cols=["zT"], train=True)
test_set = GaultoisData(target_cols=["zT"], train=False)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16 * 64, shuffle=True)


# %% [markdown]
# # Aleatoric + Epistemic Dropout


# %%
both_net = TorchDropoutModel()


# %% since we use the test set also for validation, don't use EarlyStopping or
# similar methods that peak on the validation set. decide on a number of epochs
# in advance and use the final model
both_net.fit(train_loader, test_loader)


# %%
targets, both_pred, both_std = both_net.predict(test_set)


# %%
print(f"R^2 score: {r2_score(targets, both_pred):.4g}")
al_ep_figs = plot_output(
    targets, both_pred, both_std, title="Aleatoric + Epistemic Dropout"
)


# %% [markdown]
# # Epistemic Dropout


# %%
epistemic_net = TorchDropoutModel(robust=False)


# %%
epistemic_net.fit(train_loader, test_loader)


# %%
targets, epistemic_pred, epistemic_std = epistemic_net.predict(test_set)


# %%
print(f"R^2 score: {r2_score(targets, epistemic_pred):.4g}")
ep_figs = plot_output(targets, epistemic_pred, epistemic_std, title="Epistemic Dropout")


# %% [markdown]
# # Aleatoric Dropout


# %%
aleatoric_net = TorchDropoutModel()


# %%
aleatoric_net.fit(train_loader, test_loader)


# %% n_preds=1 results in only a single dropout forward, i.e. no epistemic uncertainty
targets, aleatoric_pred, aleatoric_std = aleatoric_net.predict(test_set, n_preds=1)


# %%
print(f"R^2 score: {r2_score(targets, aleatoric_pred):.4g}")
al_figs = plot_output(targets, aleatoric_pred, aleatoric_std, title="Aleatoric Dropout")
