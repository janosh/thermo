# %%
import torch
from torch import nn
from torch.utils.data import DataLoader

from thermo.bnn.torch_dropout import GaultoisData
from thermo.plots import plot_output
from thermo.utils import ROOT


# %%
head = lambda: nn.Sequential(
    nn.Linear(20, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 1),
)


class MultiTaskMLP(nn.Module):
    def __init__(self, n_tasks):
        super().__init__()
        self.epoch = 0

        self.trunk = nn.Sequential(
            nn.Linear(146, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
        )
        self.heads = nn.ModuleList([head() for _ in range(n_tasks)])

    def forward(self, x):
        x = self.trunk(x)
        ys = [head(x).squeeze() for head in self.heads]
        return torch.stack(ys).T


# %%
target_names = ["rho", "seebeck", "kappa", "zT"]
# target_cols = ["rho", "seebeck", "kappa"]
# target_cols = ["kappa"]
train_set = GaultoisData(target_cols=target_names, train=True)
test_set = GaultoisData(target_cols=target_names, train=False)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16 * 64, shuffle=True)

n_tasks = len(target_names)
epochs = 300
report_every = 30
short_names = [s[:3] for s in target_names]

n_models = 4
models = [MultiTaskMLP(n_tasks) for _ in range(n_models)]
optims = [torch.optim.AdamW(model.parameters()) for model in models]
# with nn.L1Loss, MAE and loss are identical except the former is denormed
loss_fn = nn.L1Loss()

n_params = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
print(f"\nTotal number of trainable parameters (per model): {n_params:,}")

dummy_mae = lambda y: (y - y.mean()).abs().mean()
print(
    f"Dummy MAE: train = {dummy_mae(train_set.y):.3f}, "
    f"test = {dummy_mae(test_set.y):.3f}"
)
test_targets = train_set.denorm(test_set.y)


# %%
test_mae, test_rmse, test_preds = [], [], []

for model, optim, count in zip(models, optims, range(n_models)):
    print(f"\n\nmodel {count + 1}/{n_models}")
    total_epochs = model.epoch + epochs

    metrics = ["loss", "MAE", "RMSE", *(f"loss_{col}" for col in short_names)]
    print("epoch".ljust(10) + "".join(f"{key:<10}" for key in metrics))
    metrics = {key: [] for key in metrics}

    for epoch in range(model.epoch, total_epochs):
        for samples, targets in train_loader:
            optim.zero_grad()
            preds = model(samples)

            loss = loss_fn(preds, targets)

            loss.backward()
            optim.step()

            metrics["loss"] += [loss]
            if n_tasks > 1:
                for name, y_hat, y in zip(short_names, preds.T, targets.T):
                    metrics[f"loss_{name}"] += [loss_fn(y_hat, y)]

            preds = train_set.denorm(preds)
            targets = train_set.denorm(targets)

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

    preds = train_set.denorm(preds)
    test_preds += [preds]

    mae = (preds - test_targets).abs().mean()
    test_mae += [mae]

    rmse = (preds - test_targets).pow(2).mean().sqrt()
    test_rmse += [rmse]

    print(f"\ntest set: MAE = {mae:.3f}, RMSE = {rmse:.3f}")


test_mae, test_rmse = torch.tensor(test_mae), torch.tensor(test_rmse)
print(
    f"\ntest set: MAE = {test_mae.mean():.3f} +/- {test_mae.std():.3f}, "
    f"RMSE = {test_rmse.mean():.3f} +/- {test_rmse.std():.3f}"
)


# %%
preds = sum(test_preds) / len(test_preds)

for idx, name in enumerate(target_names):
    pred, target = preds.T[idx], test_targets.T[idx]
    fig = plot_output(target, pred, title=name)
    filename = f"{name}-ep={models[0].epoch}-tasks={','.join(short_names)}"
    fig.savefig(f"{ROOT}/results/multitask/ensemble/{filename}.pdf")
