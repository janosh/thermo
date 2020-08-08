from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from data import load_gaultois, normalize, train_test_split
from utils import ROOT


class GaultoisData(Dataset):
    def __init__(self, test_size=0.1, train=True, target_cols=None, **kwargs):
        super().__init__(**kwargs)
        features, labels = load_gaultois(target_cols=target_cols)
        labels = np.log(labels)

        features, labels = train_test_split(
            features, labels, train=train, test_size=test_size
        )

        self.features_df, [self.feature_mean, self.feature_std] = normalize(features)
        self.labels_df, [self.label_mean, self.label_std] = normalize(labels)

        self.features = torch.Tensor(self.features_df.to_numpy())
        self.labels = torch.Tensor(self.labels_df.to_numpy())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TorchDropoutModel(nn.Module):
    """
    Constructs a dropout network with aleatoric and/or epistemic uncertainty estimation.
    """

    def __init__(
        self,
        sizes=[146, 50, 25, 10],
        drop_rates=[0.5, 0.3, 0.3, 0.3],
        activations=["Tanh", "ReLU", "ReLU", "ReLU"],
        uncertainty="aleatoric_epistemic",
        optimizer=None,
        **kwargs,
    ):
        err_msg = "length mismatch in TorchDropoutModel hyperparams"
        assert len(sizes) == len(drop_rates) == len(activations), err_msg

        err_msg = f"unexpected uncertainty type: {uncertainty}"
        assert uncertainty in "aleatoric_epistemic", err_msg

        layers = []
        for n_in, n_out, act, dr in zip(sizes, sizes[1:], activations, drop_rates):
            layers.extend([nn.Linear(n_in, n_out), nn.Dropout(dr), getattr(nn, act)()])

        final = nn.Linear(sizes[-1], 1 if uncertainty == "epistemic" else 2)

        super().__init__(**kwargs)
        self.net = nn.Sequential(*layers, final)
        self.uncertainty = uncertainty

        now = f"{datetime.now():%Y-%m-%d_%H:%M:%S}"
        self.writer = SummaryWriter(f"{ROOT}/runs/torch_dropout/{uncertainty}/{now}")
        self.epoch = 0
        self.optimizer = optimizer or torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.net.forward(x).squeeze()

    def fit(self, loader, epochs=100, print_every=10, write_summary=True, **kwargs):
        self.train()

        if "aleatoric" in self.uncertainty:
            loss_fn = lambda true, pred: robust_mse(true, *pred.T)
        if "epistemic" == self.uncertainty:
            loss_fn = F.mse_loss

        print("epoch\t\ttraining loss\n")
        for epoch in range(epochs + 1):

            for samples, targets in loader:
                self.optimizer.zero_grad()
                output = self(samples)
                loss = loss_fn(targets.squeeze(), output)
                loss.backward()
                self.optimizer.step()

            if epoch % print_every == 0:
                print(f"{epoch}/{epochs}\t\t{loss:.4g}")

            if write_summary:
                self.writer.add_scalar("loss/train", loss, self.epoch)
                self.epoch += 1

    def predict(self, samples, n_preds=100):
        # We're not calling self.eval() here to ensure nn.Dropout remains active.
        self.train()

        if "epistemic" in self.uncertainty:
            preds = torch.stack([self(samples) for _ in range(n_preds)])

        elif "aleatoric" in self.uncertainty:
            preds = self(samples)

        return preds.detach().numpy()

    def write_graph(self, loader):
        samples, _ = next(iter(loader))
        self.writer.add_graph(self, samples)
        self.writer.flush()


def robust_mse(y_true, y_pred, y_log_var):
    """Mean squared error with learned loss attenuation. See eq. (7) of
    https://arxiv.org/abs/1703.04977 for details. Takes log variance for
    numerical stability. Use this version with pytorch and numpy.
    tf_dropout.py has a TF version.
    """
    loss = 0.5 * (y_true - y_pred) ** 2 / y_log_var.exp() + 0.5 * y_log_var
    return loss.mean()
