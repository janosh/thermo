# %%
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from bnn.torch_dropout import GaultoisData, TorchDropoutModel
from plots import plot_output
from utils.evaluate import r2_score

plt.rcParams["figure.figsize"] = [12, 8]


# %%
train_set = GaultoisData(target_cols=["zT"], train=True)
test_set = GaultoisData(target_cols=["zT"], train=False)
y_true = test_set.labels.values

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16 * 64, shuffle=True)


# %% [markdown]
# # Aleatoric + Epistemic Dropout


# %%
both_net = TorchDropoutModel(
    # sizes=[146, 50, 10], activations=["ReLU", "ReLU"], drop_rates=[0.5, 0.5]
)


# %%
# since we use the test set also for validation, don't use EarlyStopping or
# similar methods that peak on the validation set. decide on a number of epochs
# in advance and use the final model
both_net.fit(train_loader, test_loader)


# %%
both_pred, both_std = both_net.predict(test_set)


# %%
print(f"R^2 score: {r2_score(y_true, both_pred):.4g}")
al_ep_figs = plot_output(
    y_true, both_pred, both_std, title="Aleatoric + Epistemic Dropout"
)


# %% [markdown]
# # Epistemic Dropout


# %%
epistemic_net = TorchDropoutModel(uncertainty="epistemic")


# %%
epistemic_net.fit(train_loader, test_loader)


# %%
epistemic_pred, epistemic_var = epistemic_net.predict(test_set)


# %%
print(f"R^2 score: {r2_score(y_true, epistemic_pred):.4g}")
ep_figs = plot_output(
    y_true, epistemic_pred, epistemic_var ** 0.5, title="Epistemic Dropout"
)


# %% [markdown]
# # Aleatoric Dropout


# %%
aleatoric_net = TorchDropoutModel(uncertainty="aleatoric")


# %%
aleatoric_net.fit(train_loader, test_loader)


# %%
aleatoric_pred, aleatoric_std = aleatoric_net.predict(test_set)


# %%
print(f"R^2 score: {r2_score(y_true, aleatoric_pred):.4g}")
al_figs = plot_output(y_true, aleatoric_pred, aleatoric_std, title="Aleatoric Dropout")
