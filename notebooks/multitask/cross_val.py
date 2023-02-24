# %%
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader

from thermo.bnn.torch_dropout import Normalized
from thermo.data import dropna, load_gaultois
from thermo.plots import plot_output
from thermo.utils import ROOT


# %%
DIR = f"{ROOT}/results/multitask"

head = lambda: nn.Sequential(
    nn.Linear(20, 20),
    nn.LeakyReLU(),
    nn.Linear(20, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 1),
)


class MultiTaskMLP(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.register_buffer("epoch", 0)

        self.trunk = nn.Sequential(
            nn.Linear(146, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 20),
            nn.LeakyReLU(),
        )
        heads = [head() for _ in range(n_tasks)]
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        x = self.trunk(x)
        ys = [head(x).squeeze() for head in self.heads]
        return torch.stack(ys).T


# target_names = ["rho", "seebeck", "kappa", "zT"]
# target_names = ["rho", "seebeck", "kappa"]
target_names = ["kappa"]
features, targets = load_gaultois(target_names, drop_outliers=True)
print(f"Sparse samples: {len(targets)}")
targets, features = dropna(targets, features)
print(f"Dense samples: {len(targets)}")


# %%
n_tasks = len(target_names)
epochs = 300
report_every = epochs // 3
short_names = [s[:3] for s in target_names]

folds = 10
kfold = KFold(folds, shuffle=True, random_state=0)
models = [MultiTaskMLP(n_tasks) for _ in range(folds)]
optims = [torch.optim.AdamW(model.parameters()) for model in models]
# with nn.L1Loss, MAE and loss are identical except the former is denormed
loss_fn = nn.L1Loss()


# %%
try:
    for idx, model in enumerate(models):
        state = torch.load(f"{DIR}/{','.join(target_names)}/model_{idx}")
        model.load_state_dict(state)
except FileNotFoundError:
    pass


# %%
test_mae, test_rmse, test_preds, test_targets = [], [], [], []

for count, (train_idx, test_idx) in enumerate(kfold.split(features, targets)):
    print(f"\n\nfold {count + 1}/{folds}")
    model, optim = models[count], optims[count]
    total_epochs = model.epoch + epochs

    train_set = Normalized(features.iloc[train_idx], targets.iloc[train_idx])
    test_set = Normalized(features.iloc[test_idx], targets.iloc[test_idx])

    metrics = ["loss", "MAE", "RMSE", *(f"loss_{col}" for col in short_names)]
    print("epoch".ljust(10) + "".join(f"{key:<10}" for key in metrics))
    metrics = {key: [] for key in metrics}

    for epoch in range(model.epoch, total_epochs):
        for samples, targets in DataLoader(train_set, batch_size=32, shuffle=True):
            optim.zero_grad()
            preds = model(samples)

            loss = loss_fn(preds, targets)

            loss.backward()
            optim.step()

            metrics["loss"] += [loss]
            if n_tasks > 1:
                for name, y_hat, y in zip(short_names, preds.T, targets.T):
                    metrics[f"loss_{name}"] += [loss_fn(y_hat, y)]

            preds = test_set.denorm(preds)
            targets = test_set.denorm(targets)

            MAE = (preds - targets).abs().mean()
            metrics["MAE"] += [MAE]

            RMSE = (preds - targets).pow(2).mean().sqrt()
            metrics["RMSE"] += [RMSE]

        if epoch % report_every == 0:
            # first report has more noise since it only aggregates metrics over
            # a single epoch subsequent reports aggregate over report_every epochs
            report = f"{epoch:>3}/{total_epochs}:".ljust(10) + "".join(
                f"{sum(val) / len(val):<10.3f}" for val in metrics.values() if val
            )
            print(report)
            metrics = {key: [] for key in metrics}

        model.epoch += 1

    with torch.no_grad():
        preds = model(test_set.X)

    preds = test_set.denorm(preds)
    targets = test_set.denorm(test_set.y)

    test_preds += [preds]
    test_targets += [targets]

    mae = (preds - targets).abs().mean(0)
    test_mae += [mae]

    rmse = (preds - targets).pow(2).mean(0).sqrt()
    test_rmse += [rmse]

    print(f"\ntest set: avg. MAE = {mae.mean():.3f}, avg. RMSE = {rmse.mean():.3f}")

mae_df = pd.DataFrame(test_mae, columns=target_names).astype(float)
rmse_df = pd.DataFrame(test_rmse, columns=target_names).astype(float)


# %%
print(models[0])
for idx, model in enumerate(models):
    os.makedirs(f"{DIR}/{','.join(target_names)}", exist_ok=True)
    torch.save(model.state_dict(), f"{DIR}/{','.join(target_names)}/model_{idx}")


# %%
print(f"Using {folds}-fold cross validation")

count_params = lambda net: sum(p.numel() for p in net.parameters() if p.requires_grad)

n_params = count_params(models[0])
print(f"\nTrainable params per model: {n_params:,}")

head_params = count_params(models[0].heads[0])
print(f"\nTrainable params per head: {head_params:,}")
print(f"\ntrunk_params / head_params: {(n_params - head_params) / head_params:.1f}")


# %%
mean_pm_std = lambda vals: f"{vals.mean():.3g} +/- {vals.std():.3g}"
std_over_mean = lambda vals: f"(std/mean = {vals.std() / vals.mean():.3g})"

print("\nTest MAE")
for key, col in mae_df.iteritems():
    print(f"{key:<7} = {mean_pm_std(col).ljust(22)} {std_over_mean(col)}")

if n_tasks > 1:
    print(f"{'all':<7} = {mae_df.mean().mean():.3g} +/- {mae_df.std().mean():.3g}")

print("\nTest RMSE")
for key, col in rmse_df.iteritems():
    print(f"{key:<7} = {mean_pm_std(col).ljust(22)} {std_over_mean(col)}")

if n_tasks > 1:
    print(f"{'all':<7} = {rmse_df.mean().mean():.3g} +/- {rmse_df.std().mean():.3g}")


dummy_mae = lambda y: (y - y.mean()).abs().mean()
dummy_rmse = lambda y: (y - y.mean()).pow(2).mean() ** 0.5

print("\nDummy MAE".ljust(25) + "Dummy RMSE")
for key, col in targets.iteritems():
    print(
        f"{key:<7} = {dummy_mae(col):.3g}".ljust(24)
        + f"{key:<7} = {dummy_rmse(col):.3g}"
    )

if n_tasks > 1:
    print(
        f"{'all':<7} = {dummy_mae(targets).mean():.3g}".ljust(24)
        + f"{'all':<7} = {dummy_rmse(targets).mean():.3g}"
    )


# %%
for idx, name in enumerate(target_names):
    pred, target = test_preds[1].T[idx], test_targets[1].T[idx]
    pred = pred[target < np.percentile(target, 90)]
    target = target[target < np.percentile(target, 90)]
    fig = plot_output(target, pred, title=name)
    filename = f"{name}-ep={models[0].epoch}-tasks={','.join(short_names)}"
    fig.savefig(f"{DIR}/cross_val/{filename}.pdf")
