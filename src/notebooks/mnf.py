"""
MNF vs dropout vs RF, all with Magpie features

This notebook compares performance of Multiplicative Normalizing Flow (MNF)
against random forest (RF), both using Magpie features.
"""


# %%
from datetime import datetime

import tensorflow as tf
from tf_mnf.models import MNFFeedForward
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from bnn.torch_dropout import GaultoisData, TorchDropoutModel
from data import dropna, load_gaultois, normalize, train_test_split
from rf import rf_predict
from utils import ROOT
from utils.evaluate import plot_output

# %%
features, labels = load_gaultois()
labels, features = dropna(labels, features)

features, [X_mean, X_std] = normalize(features)
labels, [y_mean, y_std] = normalize(labels)

[X_train, y_train], [X_test, y_test] = train_test_split(features, labels)


# %%
mnf_model = MNFFeedForward(layer_dims=(100, 50, 10, 1))
adam = tf.optimizers.Adam()
batch_size = 32


# %%
def loss_fn(y_true, y_pred):
    mse_loss = tf.metrics.mse(y_true, y_pred)
    kl_loss = mnf_model.kl_div() / (2 * batch_size)
    return mse_loss + kl_loss


# %%
mnf_model.compile(loss=loss_fn, optimizer=adam, metrics=["mse"])


# %%
cb = tf.keras.callbacks.TensorBoard(
    log_dir=ROOT + f"/runs/mnf-bnn/{datetime.now():%m-%d_%H:%M:%S}"
)
mnf_hist = mnf_model.fit(
    X_train.values.astype("float32"),
    y_train.zT.values.astype("float32"),
    validation_split=0.1,
    callbacks=[cb],
    batch_size=batch_size,
    epochs=100,
)


# %%
def predict_mnf_lenet(X, n_samples=50):
    preds = []
    for _ in tqdm(range(n_samples), desc="Sampling"):
        preds.append(mnf_model(X, training=False))
    return tf.squeeze(preds)


# %%
mnf_preds = predict_mnf_lenet(X_test.values.astype("float32")).numpy()


# %%
plot_output(y_test.zT.values, mnf_preds.mean(0), mnf_preds.std(0), title="MNF")


# %%
rf_pred, rf_var, _ = rf_predict(X_train, y_train.zT, X_test, y_test.zT)


# %%
plot_output(y_test.zT.values, rf_pred, rf_var * 0.5, title="RF")


# %%
do_model = TorchDropoutModel()


# %%

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
plot_output(test_set.labels.values, do_pred, do_var * 0.5, title="DO")
