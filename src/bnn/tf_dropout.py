import numpy as np
import tensorflow as tf

from utils import plots
from utils.decorators import timed


class Dropout(tf.keras.layers.Layer):
    """Always-on dropout layer, i.e. does not respect the training flag
    set to true by model.fit but false by model.predict.
    Unlike tf.keras.layers.Dropout, this layer does not return input
    unchanged if training=true, but always randomly drops a fraction specified
    by self.size of the input nodes.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, self.rate)

    def get_config(self):
        """enables model.save and restoration through tf.keras.models.load_model"""
        config = super().get_config()
        config["rate"] = self.rate
        return config


def build_model(n_features, hyperparams=None, uncertainty="aleatoric_epistemic"):
    """
    Build fully-connected neural network with aleatoric and/or epistemic
    uncertainty estimation.

    n_features (int): number of features (columns in the matrix X)
    """
    if not hyperparams:
        nodes = (100, 50, 25, 10)
        dropout_rates = (0.5, 0.3, 0.3, 0.3)
        activations = ("tanh", "relu", "relu", "relu")
        hyperparams = zip(nodes, dropout_rates, activations)

    inputs = head = tf.keras.Input(shape=(n_features,), name="input")

    for nodes, drop_rate, act_func in hyperparams:
        head = tf.keras.layers.Dense(nodes, activation=act_func)(head)
        head = Dropout(drop_rate)(head)

    if uncertainty == "epistemic":
        outputs = tf.keras.layers.Dense(1, activation="linear", name="output")(head)
        loss = "mse"
    elif "aleatoric" in uncertainty:
        # Standard predictive (mean) output. Same as with epistemic uncertainty.
        pred_output = tf.keras.layers.Dense(1, activation="linear", name="pred_output")(
            head
        )
        # Data-dependent uncertainty output.
        var_output = tf.keras.layers.Dense(1, activation="linear", name="var_output")(
            head
        )
        outputs = [pred_output, var_output]

        pred_loss = lambda y_true, y_pred: robust_mse(y_true, y_pred, var_output)
        var_loss = lambda y_true, y_log_var: robust_mse(y_true, pred_output, y_log_var)
        loss = {"pred_output": pred_loss, "var_output": var_loss}
    else:
        raise ValueError(f"build_model received unexpected uncertainty {uncertainty}")

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name=uncertainty + "_dropout_model"
    )
    model.compile(optimizer="adam", loss=loss)

    model.uncertainty = uncertainty
    return model


@timed
def train_model(model, X_train, y_train, epochs=500, **kwargs):
    if "aleatoric" in model.uncertainty:
        history = model.fit(
            X_train,
            {"pred_output": y_train, "var_output": y_train},
            epochs=epochs,
            validation_split=0.1,
            **kwargs,
        )
    elif model.uncertainty == "epistemic":
        history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.1)
    return history.history


@timed
def predict(model, X_test, n_preds=500):
    """
    perform n_preds Monte Carlo predictions (i.e. with dropout)
    save and return predictive mean and total uncertainty
    model: pre-trained Keras model
    X_test: features tensor
    n_preds: number of predictions (with dropout)
    """
    if model.uncertainty == "aleatoric":
        y_pred, y_log_var = tf.squeeze(model.predict(X_test))
        y_var = tf.exp(y_log_var)
    else:
        output = tf.squeeze([model.predict(X_test) for _ in range(n_preds)])
        if model.uncertainty == "epistemic":
            y_pred, y_var = tf.nn.moments(output, axes=0)
        if model.uncertainty == "aleatoric_epistemic":
            # compute predictive mean and total uncertainty of n_preds forward passes
            preds, log_vars = tf.unstack(tf.squeeze(output), axis=1)
            # preds, log_vars = tf.unstack(tf.squeeze(output), axis=-1)
            y_pred, y_var_epist = tf.nn.moments(preds, axes=0)
            y_var_aleat = tf.reduce_mean(tf.exp(log_vars), axis=0)
            # total variance given by sum of aleatoric and epistemic contribution
            y_var = y_var_epist + y_var_aleat

    return y_pred.numpy(), y_var.numpy()


def do_predict(X_train, y_train, X_test, y_test, **kwargs):
    defaults = [
        ("hyperparams", None),
        ("epochs", 500),
        ("n_preds", 500),
        ("cbs", []),
        ("uncertainty", "aleatoric_epistemic"),
    ]
    hyperparams, epochs, n_preds, cbs, uncertainty = [
        kwargs.pop(*kw) for kw in defaults
    ]

    model = build_model(
        X_train.shape[1], uncertainty=uncertainty, hyperparams=hyperparams
    )
    history = train_model(
        model, X_train, y_train, callbacks=cbs, epochs=epochs, verbose=0, **kwargs
    )
    y_pred, y_var = predict(model, X_test, n_preds=n_preds)
    return y_pred, y_var, history, model


def plot_model(model, path):
    tf.keras.utils.plot_model(
        model,
        to_file=path + "model.pdf",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",  # TB or LR
    )


def load_model(path, filename="model.h5"):
    restored_model = tf.keras.models.load_model(
        path + filename, custom_objects={"Dropout": Dropout}
    )
    return restored_model


def save_loss_history(loss_history, path):
    if hasattr(loss_history, "val_loss"):
        loss_history["validation loss"] = loss_history.pop("val_loss")
        loss_history["training loss"] = loss_history.pop("loss")

    plots.loss_history(loss_history)
    header, cols = zip(*loss_history.items())
    header, cols = ", ".join(header), np.transpose(cols)
    to_file = path + "loss_history.csv"
    np.savetxt(to_file, cols, header=header, delimiter=", ", fmt="%5g")


def robust_mse(y_true, y_pred, y_log_var):
    """See torch_dropout.py for docstring.
    """
    loss = 0.5 * tf.square(y_true - y_pred) * tf.exp(-y_log_var) + 0.5 * y_log_var
    return tf.reduce_mean(loss)
