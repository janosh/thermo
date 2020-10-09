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
        err_msg = "length mismatch in hypers"
        assert len(sizes) - 1 == len(drop_rates) == len(activations), err_msg

        valid_uncert = ["aleatoric", "epistemic", "aleatoric_epistemic"]
        assert uncertainty in valid_uncert, f"unexpected uncertainty: {uncertainty}"

        super().__init__(**kwargs)
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

        now = f"{datetime.now():%Y-%m-%d_%H:%M:%S}"
        self.writer = SummaryWriter(f"{ROOT}/runs/torch_dropout/{uncertainty}/{now}")
        self.epoch = 0
        self.optimizer = optimizer or torch.optim.Adam(self.parameters())

    @interruptable
    def fit(
        self, loader, epochs=100, print_every=10, write_loss=True, write_params=False
    ):
        self.train()

        if "aleatoric" in self.uncertainty:
            loss_fn = lambda true, pred: robust_mse(true, *pred.T)
        else:
            loss_fn = F.mse_loss

        self.epoch += epochs
        print("epoch\t\ttraining loss\n")
        for epoch in range(self.epoch - epochs, self.epoch + 1):
            loss = 0

            for samples, targets in loader:
                self.optimizer.zero_grad()
                output = self(samples)
                loss = loss_fn(targets.squeeze(), output)
                loss.backward()
                self.optimizer.step()

            if epoch % print_every == 0:
                print(f"{epoch}/{self.epoch}\t\t{loss:.4g}")

            if write_loss:
                self.writer.add_scalar("loss/train", loss, self.epoch)
            if write_params:
                self.write_params(epoch)

    def predict(self, samples, n_preds=100, raw=False):
        # We're not calling self.eval() here to ensure nn.Dropout remains active.
        self.train()

        if self.uncertainty == "aleatoric":
            y_pred, y_log_var = self(samples).detach().numpy().T
            y_var = np.exp(y_log_var)
        else:
            output = np.squeeze(
                [self(samples).detach().numpy() for _ in range(n_preds)]
            )
            if raw:
                return output
            if self.uncertainty == "epistemic":
                y_pred, y_var = output.mean(0), output.var(0)
            else:
                preds, log_vars = np.rollaxis(output, -1)
                y_pred, y_var_epist = preds.mean(0), preds.var(0)
                y_var_aleat = np.exp(log_vars).mean(0)
                # total variance given by sum of aleatoric and epistemic contribution
                y_var = y_var_epist + y_var_aleat

        return y_pred, y_var

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


def robust_mse(y_true, y_pred, y_log_var):
    """Mean squared error with learned loss attenuation. See eq. (7) of
    https://arxiv.org/abs/1703.04977 for details. Takes log variance for
    numerical stability. Use this version with pytorch and numpy.
    tf_dropout.py has a TF version.
    """
    loss = 0.5 * (y_true - y_pred) ** 2 / y_log_var.exp() + 0.5 * y_log_var
    return loss.mean()
