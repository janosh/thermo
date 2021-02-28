"""
MNF vs RF (vs. dropout) using Magpie and AMM features

This notebook compares performance of Multiplicative Normalizing Flow (MNF)
against random forest (RF) (and dropout), testing first Magpie, then
automatminer features.
"""


# %%
import os

import tensorflow as tf
from tf_mnf.models import MNFFeedForward
from torch.utils.data.dataloader import DataLoader

from thermo.bnn.torch_dropout import GaultoisData, TorchDropoutModel
from thermo.data import dropna, load_gaultois, normalize, train_test_split
from thermo.plots import plot_output
from thermo.rf import rf_predict
from thermo.utils import ROOT, amm

# %%
SAVE_TO = ROOT + "/results/mnf/"
os.makedirs(SAVE_TO, exist_ok=True)
tf.random.set_seed(0)


# %%
magpie_fea, targets = load_gaultois(target_cols=["formula", "zT", "T"])
targets, magpie_fea = dropna(targets, magpie_fea)
zT = targets.pop("zT")


# %%
mat_pipe = amm.MatPipe.load(ROOT + "/results/amm/amm_vs_magpie/mat.pipe")
amm_fea = amm.featurize(mat_pipe, targets.rename(columns={"formula": "composition"}))


# %%
magpie_fea, [mp_mean, mp_std] = normalize(magpie_fea)
amm_fea, [amm_mean, amm_std] = normalize(amm_fea)

train_set, test_set = train_test_split(magpie_fea, amm_fea, targets, zT)

mp_train, amm_train, y_train, zT_train = train_set
mp_test, amm_test, y_test, zT_test = test_set


# %% [markdown]
# # Multiplicative Normalizing Flow


# %%
mp_mnf_model = MNFFeedForward(layer_sizes=(100, 50, 10, 1))
amm_mnf_model = MNFFeedForward(layer_sizes=(100, 50, 10, 1))
adam = tf.optimizers.Adam()
n_samples = len(mp_train)


# %%
def loss_factory(model):
    def loss_fn(y_true, y_pred):
        mse = tf.metrics.mse(y_true, y_pred)
        # KL div is reweighted such that it's applied once per epoch
        kl_loss = model.kl_div() / n_samples * 1e-3
        return mse + kl_loss

    return loss_fn


# %%
mp_mnf_model.compile(adam, loss_factory(mp_mnf_model), metrics=["mse"])

amm_mnf_model.compile(adam, loss_factory(amm_mnf_model), metrics=["mse"])


# %%
stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_mse", patience=50, restore_best_weights=True
)


# %%
mp_mnf_hist = mp_mnf_model.fit(
    mp_train,
    zT_train,
    validation_data=(mp_test, zT_test),
    batch_size=32,
    epochs=200,
    verbose=0,
    callbacks=[stop_early],
)


# %%
amm_mnf_hist = amm_mnf_model.fit(
    amm_train,
    zT_train,
    validation_data=(amm_test, zT_test),
    batch_size=32,
    epochs=200,
    verbose=0,
    callbacks=[stop_early],
)


# %%
n_preds = 500
mp_mnf_preds = mp_mnf_model(mp_test.values.repeat(n_preds, axis=0))
mp_mnf_preds = mp_mnf_preds.numpy().reshape(-1, n_preds).T


# %%
amm_mnf_preds = amm_mnf_model(amm_test.values.repeat(n_preds, axis=0))
amm_mnf_preds = amm_mnf_preds.numpy().reshape(-1, n_preds).T


# %%
mp_mnf_figs = plot_output(
    zT_test.values, mp_mnf_preds.mean(0), mp_mnf_preds.std(0), title="Magpie + MNF"
)


# %%
amm_mnf_figs = plot_output(
    zT_test.values, amm_mnf_preds.mean(0), amm_mnf_preds.std(0), title="AMM + MNF"
)


# %%
fig_titles = ["true_vs_pred", "decay_by_std", "abs_err_vs_std"]
for fig, name in zip(mp_mnf_figs, fig_titles):
    fig.savefig(SAVE_TO + name + "-mp-mnf.pdf", bbox_inches="tight")


# %% [markdown]
# # Random Forest


# %%
mp_rf_pred, mp_rf_var, _ = rf_predict(mp_train, zT_train, mp_test)


# %%
figs_rf = plot_output(zT_test.values, mp_rf_pred, mp_rf_var ** 0.5, title="Magpie + RF")


# %%
amm_rf_pred, amm_rf_var, _ = rf_predict(amm_train, zT_train, amm_test)


# %%
figs_rf = plot_output(zT_test.values, amm_rf_pred, amm_rf_var ** 0.5, title="AMM + RF")


# %%
for fig, name in zip(figs_rf, fig_titles):
    fig.savefig(SAVE_TO + name + "-mp-rf.pdf", bbox_inches="tight")


# %% [markdown]
# # PyTorch Dropout


# %%
do_model = TorchDropoutModel()

train_set = GaultoisData(target_cols=["zT"], train=True)
test_set = GaultoisData(target_cols=["zT"], train=False)

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)


# %%
do_model.fit(train_loader)


# %%
do_pred, do_var = do_model.predict(test_set.X)


# %%
do_pred = do_pred * test_set.y_std + test_set.y_mean
do_var *= test_set.y_std ** 2


# %%
plot_output(test_set.targets.values, do_pred, do_var * 0.5, title="Dropout")
