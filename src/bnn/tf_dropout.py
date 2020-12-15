import tensorflow as tf

from utils.decorators import timed


class Dropout(tf.keras.layers.Layer):
    """Always-on dropout layer. Disregards the training flag flag set
    to true in model.fit() and false in model.predict(). Unlike
    tf.keras.layers.Dropout, this layer does not return its input
    unchanged if training=false, but always randomly drops input nodes
    with probability self.rate.
    """

    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs):
        return tf.nn.dropout(inputs, self.rate)

    def get_config(self):
        """Enables model.save and restoration through tf.keras.models.load_model."""
        config = super().get_config()
        config["rate"] = self.rate
        return config


class TFDropoutModel(tf.keras.Model):
    """Build fully-connected dropout neural network with aleatoric and/or epistemic
    uncertainty estimation.
    """

    def __init__(
        self,
        n_features,  # (int) number of features (columns in the input matrix X)
        h_sizes=(100, 50, 25, 10),  # (List[int]) size of hidden layers
        drop_rates=(0.5, 0.3, 0.3, 0.3),  # dropout rates after each hidden layer
        activations=("tanh", "relu", "relu", "relu"),
        uncertainty="aleatoric_epistemic",
        optim="adam",
    ):
        assert (
            len(h_sizes) == len(drop_rates) == len(activations)
        ), "length mismatch in hypers"

        valid_uncert = ["aleatoric", "epistemic", "aleatoric_epistemic"]
        assert uncertainty in valid_uncert, f"unexpected uncertainty: {uncertainty}"

        self.uncertainty = uncertainty

        inputs = head = tf.keras.Input(shape=[n_features])

        for size, drop_rate, act_func in zip(h_sizes, drop_rates, activations):
            head = tf.keras.layers.Dense(size, activation=act_func)(head)
            head = Dropout(drop_rate)(head)

        # If "aleatoric" in uncertainty, first node gives predictive mean (same as with
        # epistemic uncertainty), second node gives data-dependent/heteroscedastic
        # uncertainty (variance).
        outputs = tf.keras.layers.Dense(
            1 if uncertainty == "epistemic" else 2, activation="linear"
        )(head)
        super().__init__(inputs, outputs, name=uncertainty + "_dropout_model")

        aleatoric_loss = lambda y_true, output: robust_mse(
            y_true, *tf.unstack(output, axis=-1)
        )
        self.compile(
            optimizer=optim,
            loss="mse" if uncertainty == "epistemic" else aleatoric_loss,
        )


@timed
def predict(model, X_test, n_preds=100):
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
            preds, log_vars = tf.unstack(output, axis=-1)
            y_pred, y_var_epist = tf.nn.moments(preds, axes=0)
            y_var_aleat = tf.reduce_mean(tf.exp(log_vars), axis=0)
            # total variance given by sum of aleatoric and epistemic contribution
            y_var = y_var_epist + y_var_aleat

    return y_pred.numpy(), y_var.numpy()


def do_predict(X_train, y_train, X_test, y_test, **kwargs):
    defaults = [
        ("epochs", 100),
        ("n_preds", 100),
        ("cbs", []),
        ("uncertainty", "aleatoric_epistemic"),
    ]
    epochs, n_preds, cbs, uncertainty = [
        kwargs.pop(key, default) for key, default in defaults
    ]

    model = TFDropoutModel(X_train.shape[1], uncertainty=uncertainty)
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_split=0.1, callbacks=cbs, **kwargs
    )
    y_pred, y_var = predict(model, X_test, n_preds=n_preds)
    return y_pred, y_var, history.history, model


def plot_model(model, path):
    tf.keras.utils.plot_model(
        model,
        to_file=path + "model.pdf",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",  # TB or LR
    )


def load_model(path, filename="model.h5"):
    return tf.keras.models.load_model(
        path + filename, custom_objects={"Dropout": Dropout}
    )


def robust_mse(y_true, y_pred, y_log_var):
    """See torch_dropout.py for docstring."""
    loss = 0.5 * tf.square(y_true - y_pred) * tf.exp(-y_log_var) + 0.5 * y_log_var
    return tf.reduce_mean(loss)
