from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from data import dropna, load_gaultois, normalize, train_test_split
from utils import ROOT
from utils.decorators import interruptable


class GaultoisData(Dataset):
    def __init__(self, test_size=0.1, train=True, target_cols=None, **kwargs):
        super().__init__(**kwargs)

        features, labels = load_gaultois(target_cols=target_cols)
        labels, features = dropna(labels.squeeze(), features)

        self.features, self.labels = train_test_split(
            features, labels, train=train, test_size=test_size
        )

        X, [self.X_mean, self.X_std] = normalize(self.features)
        y, [self.y_mean, self.y_std] = normalize(self.labels)

        self.X = torch.Tensor(X.to_numpy())
        self.y = torch.Tensor(y.to_numpy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def denorm(self, tensor, is_std=False):
        """ Revert z-scoring/normalization. """
        if is_std:
            return tensor * self.y_std
        return tensor * self.y_std + self.y_mean


def robust_l1_loss(targets, preds, log_stds):
    """Robust L1 loss using a Lorentzian prior.
    Allows for aleatoric uncertainty estimation.
    """
    loss = 2 ** 0.5 * (preds - targets).abs() / log_stds.exp() + log_stds
    return loss.mean()


def robust_l2_loss(targets, preds, log_stds):
    """Robust L2 loss using a gaussian prior.
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
        activations=["Tanh", "ReLU", "ReLU", "ReLU"],
        uncertainty="aleatoric_epistemic",
        optimizer=None,
        **kwargs,
    ):
        err_msg = "length mismatch in hyperparameters"
        assert len(sizes) - 1 == len(drop_rates) == len(activations), err_msg

        valid_uncert = ["aleatoric", "epistemic", "aleatoric_epistemic"]
        assert uncertainty in valid_uncert, f"unexpected uncertainty: {uncertainty}"

        super().__init__(**kwargs)

        # build network
        for idx, [n_in, n_out, act, dr] in enumerate(
            zip(sizes, sizes[1:], activations, drop_rates), 1
        ):
            self.add_module(f"{idx}a", nn.Linear(n_in, n_out))
            self.add_module(f"{idx}b", nn.Dropout(dr))
            self.add_module(f"{idx}c", getattr(nn, act)())

        self.add_module(
            "final", nn.Linear(sizes[-1], 1 if uncertainty == "epistemic" else 2)
        )

        self.uncertainty = uncertainty
        self.robust = "aleatoric" in uncertainty  # whether to use robust loss function
        self.epochs = 0

        now = f"{datetime.now():%Y-%m-%d_%H:%M:%S}"
        self.writer = SummaryWriter(f"{ROOT}/runs/torch_dropout/{uncertainty}/{now}")

        metrics = ["loss", "mae", "rmse"] + (["std_al"] if self.robust else [])
        prefixes = ["training/", "validation/"]
        # order is important for logging, iterate over prefixes
        # first to separate training from validation metrics
        metrics = [prefix + key for prefix in prefixes for key in metrics]
        self.metrics = {key: [] for key in metrics}

        self.optimizer = optimizer or torch.optim.AdamW(self.parameters())
        self.loss_fn = (
            lambda y, out: robust_l1_loss(y, *out.T) if self.robust else F.l1_loss
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

    @interruptable
    def fit(
        self,
        loader,
        val_loader=None,
        epochs=100,
        print_every=10,
        log=True,
        write_params=False,
    ):
        self.train()
        # callable to revert z-scoring of targets and predictions to real units
        denorm = loader.dataset.denorm
        metrics = self.metrics

        cols = "epoch\t " + " ".join(f"{itm.split('/')[1]:<10}" for itm in metrics)
        if val_loader is not None:
            cols = f"\t\t\tTraining \t\t\t Validation\n{cols} "
        print(cols)

        for epoch in range(self.epochs, self.epochs + epochs + 1):

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
                report = f"{epoch}/{self.epochs}\t " + " ".join(
                    f"{val:<10.4f}" for val in metrics.values()
                )
                print(report)

            if write_params:
                self.write_params(epoch)

            self.epochs += 1

    def predict(self, data, n_preds=100, raw=False):
        # We're not calling self.eval() here to ensure nn.Dropout remains active.
        self.train()

        if self.uncertainty == "aleatoric":
            preds, log_stds = self(data.X).detach().numpy().T
            stds = np.exp(log_stds)
        else:  # uncertainty includes "epistemic"
            output = np.squeeze([self(data.X).detach().numpy() for _ in range(n_preds)])
            if raw:
                return output
            if self.uncertainty == "epistemic":
                preds, stds = output.mean(0), output.std(0)
            else:
                preds, log_stds_al = np.rollaxis(output, -1)
                preds, std_ep = preds.mean(0), preds.std(0)
                stds_al = np.exp(log_stds_al).mean(0)
                # total variance given by sum of aleatoric and epistemic contribution
                stds = (std_ep ** 2 + stds_al ** 2) ** 0.5
        preds = data.denorm(preds)
        stds = data.denorm(stds, is_std=True)

        return preds, stds

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
