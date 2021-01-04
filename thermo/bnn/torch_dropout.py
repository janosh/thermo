from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from thermo.data import dropna, load_gaultois, normalize, train_test_split
from thermo.utils import ROOT


class Normalized(Dataset):
    def __init__(self, features, targets):
        super().__init__()
        self.features, self.targets = features, targets

        X, [X_mean, X_std] = normalize(features)
        self.X_mean, self.X_std = torch.tensor(X_mean), torch.tensor(X_std)

        # transpose targets to make target_cols the first dimension
        y, [y_mean, y_std] = normalize(targets)
        self.y_mean, self.y_std = torch.tensor(y_mean), torch.tensor(y_std)

        self.X = torch.tensor(X.to_numpy())
        self.y = torch.tensor(y.to_numpy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def denorm(self, tensor, is_std=False):
        """ Revert z-scoring/normalization. """
        if is_std:
            return tensor * self.y_std
        return tensor * self.y_std + self.y_mean

    def denorm_X(self, tensor, is_std=False):
        """ Revert z-scoring/normalization. """
        if is_std:
            return tensor * self.X_std
        return tensor * self.X_std + self.X_mean


class GaultoisData(Normalized):
    def __init__(self, test_size=0.1, train=True, target_cols=None):

        features, targets = load_gaultois(target_cols=target_cols)
        targets, features = dropna(targets, features)

        features, targets = train_test_split(
            features, targets, train=train, test_size=test_size
        )
        super().__init__(features, targets)


def robust_l1_loss(targets, preds, log_stds):
    """Robust L1 loss using a Lorentzian prior.
    Allows for aleatoric uncertainty estimation.
    """
    loss = 2 ** 0.5 * (preds - targets).abs() / log_stds.exp() + log_stds
    return loss.mean()


def robust_l2_loss(targets, preds, log_stds):
    """Robust L2 loss using a Gaussian prior.
    Allows for aleatoric uncertainty estimation.
    """
    loss = 0.5 * (preds - targets) ** 2 / (2 * log_stds).exp() + log_stds
    return loss.mean()


class TorchDropoutModel(nn.Sequential):
    """
    Constructs a dropout network with aleatoric and/or epistemic uncertainty estimation.
    """

    def __init__(
        self,
        sizes=[146, 100, 50, 25, 10],
        drop_rates=[0.5, 0.3, 0.3, 0.3],
        activations=["LeakyReLU"] * 4,
        robust=True,  # whether to use robust loss function
        optimizer=None,
        **kwargs,
    ):
        err_msg = "length mismatch in hyperparameters"
        assert len(sizes) - 1 == len(drop_rates) == len(activations), err_msg

        super().__init__(**kwargs)

        # build network
        for idx, [n_in, n_out, act, dr] in enumerate(
            zip(sizes, sizes[1:], activations, drop_rates), 1
        ):
            self.add_module(f"{idx}a", nn.Linear(n_in, n_out))
            self.add_module(f"{idx}b", nn.Dropout(dr))
            self.add_module(f"{idx}c", getattr(nn, act)())

        self.add_module("final", nn.Linear(sizes[-1], 2 if robust else 1))

        self.robust = robust
        self.epochs = 0

        now = f"{datetime.now():%Y-%m-%d_%H:%M:%S}"
        self.writer = SummaryWriter(
            f"{ROOT}/runs/torch_dropout{'_robust' if robust else ''}/{now}"
        )

        metrics = ["loss", "mae", "rmse"] + (["std_al"] if robust else [])
        prefixes = ["training/", "validation/"]
        # order is important for logging, iterate over prefixes
        # first to separate training from validation metrics
        metrics = [prefix + key for prefix in prefixes for key in metrics]
        self.metrics = {key: [] for key in metrics}

        self.optimizer = optimizer or torch.optim.AdamW(self.parameters())
        self.loss_fn = lambda y, out: (
            robust_l1_loss(y, *out.T) if robust else nn.L1Loss()(y, out)
        )

    @torch.no_grad()
    def write_metrics(self, targets, output, denorm, prefix):
        """ After an epoch, save evaluation metrics to a dict. """

        output, targets = torch.cat(output), torch.cat(targets)
        loss = self.loss_fn(targets, output)

        if self.robust:
            output, log_std = output.T
            self.metrics[prefix + "/std_al"] = log_std.exp().mean()

        targets, output = denorm(targets), denorm(output)

        self.metrics[prefix + "/loss"] = loss
        self.metrics[prefix + "/mae"] = (output - targets).abs().mean()
        self.metrics[prefix + "/rmse"] = (output - targets).pow(2).mean().sqrt()

    def fit(
        self, loader, val_loader=None, epochs=100, print_every=10, log=True, cbs=[]
    ):
        self.train()
        # callable to revert z-scoring of targets and predictions to real units
        denorm = loader.dataset.denorm
        metrics = self.metrics
        epochs += self.epochs

        cols = "epoch\t " + "".join(f"{itm.split('/')[1]:<10}" for itm in metrics)
        if val_loader is not None:
            ljust = 10 * len(metrics) // 2
            heading = " " * 9 + "Training".ljust(ljust) + "Validation".ljust(ljust)
            cols = f"{heading}\n{cols}"
        print(cols)

        for epoch in range(self.epochs, epochs):

            targets, outputs = [], []

            for samples, target in loader:
                self.optimizer.zero_grad()
                output = self(samples)
                loss = self.loss_fn(target, output)
                loss.backward()
                self.optimizer.step()

                outputs.append(output)
                targets.append(target)

            self.write_metrics(targets, outputs, denorm, "training")

            if val_loader is not None:
                with torch.no_grad():
                    output, targets = zip(*[[self(x), y] for x, y in loader])
                    self.write_metrics(targets, output, denorm, "validation")

            if log:
                for key, val in metrics.items():
                    self.writer.add_scalar(key, val, self.epochs)

            if epoch % print_every == 0:
                report = f"{epoch + 1}/{epochs}\t " + "".join(
                    f"{val:<10.4f}" for val in metrics.values()
                )
                print(report)

            for cb in cbs:  # list of callbacks
                cb()  # e.g. self.write_params() tracks model weights while training

            self.epochs += 1

    @torch.no_grad()
    def predict(self, data, n_preds=100, raw=False):
        # calling self.train() to ensure nn.Dropout remains active
        self.train()
        output = torch.stack([self(data.X) for _ in range(n_preds)]).squeeze()

        if raw:
            return output

        if self.robust:
            pred, log_stds_al = output.transpose(0, -1)
            std_al = log_stds_al.exp()

            if n_preds > 1:
                # compute mean and std over last axis of repeated dropout forwards
                pred, std_ep = pred.mean(-1), pred.std(-1)
                std_al = std_al.mean(-1)
                # total variance given by sum of aleatoric and epistemic contribution
                std = (std_ep ** 2 + std_al ** 2) ** 0.5
            else:
                std = std_al
        else:
            pred, std = output.mean(0), output.std(0)

        pred = data.denorm(pred)
        std = data.denorm(std, is_std=True)

        return data.targets.values, pred.numpy(), std.numpy()

    def write_graph(self, loader):
        samples, _ = next(iter(loader))
        self.writer.add_graph(self, samples)
        self.writer.flush()

    def write_params(self, epoch, write_grads=False):
        for name, child in self.named_children():
            for kind, param in child.named_parameters():
                self.writer.add_histogram(f"{name}_{kind}", param, epoch)
                if write_grads:
                    self.writer.add_histogram(f"{name}_{kind}_grad", param.grad, epoch)
