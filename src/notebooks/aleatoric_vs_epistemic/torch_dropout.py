# %%
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from bnn.torch_dropout import GaultoisData, TorchDropoutModel
from utils.evaluate import plot_output, r2_score

plt.rcParams["figure.figsize"] = [12, 8]


# %%
train_set = GaultoisData(target_cols=["rho"], train=True)
test_set = GaultoisData(target_cols=["rho"], train=False)
y_true = test_set.y.numpy()

train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)


# %% [markdown]
# # Aleatoric + Epistemic Dropout


# %%
both_net = TorchDropoutModel()


# %%
both_net.fit(train_loader)


# %%
both_net.write_graph(train_loader)


# %%
both_pred, both_var = both_net.predict(test_set.X)


# %%
print(f"R^2 score: {r2_score(y_true, both_pred):.4g}")
plot_output(y_true, both_pred, both_var ** 0.5, title="Aleatoric + Epistemic Dropout")


# %% [markdown]
# # Epistemic Dropout


# %%
epistemic_net = TorchDropoutModel(uncertainty="epistemic")


# %%
epistemic_net.fit(train_loader)


# %%
epistemic_pred, epistemic_var = epistemic_net.predict(test_set.X)


# %%
print(f"R^2 score: {r2_score(y_true, epistemic_pred):.4g}")
plot_output(y_true, epistemic_pred, epistemic_var ** 0.5, title="Epistemic Dropout")


# %% [markdown]
# # Aleatoric Dropout


# %%
aleatoric_net = TorchDropoutModel(uncertainty="aleatoric")


# %%
aleatoric_net.fit(train_loader)


# %%
aleatoric_pred, aleatoric_std = aleatoric_net.predict(test_set.X)


# %%
print(f"R^2 score: {r2_score(y_true, aleatoric_pred):.4g}")
plot_output(y_true, aleatoric_pred, aleatoric_std, title="Aleatoric Dropout")
