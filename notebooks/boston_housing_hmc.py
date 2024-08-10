"""This notebook essentially runs an end-to-end test comparing RF vs MAP NN vs HMC
NN performance on the simple Boston housing dataset.
"""

# %%
import tensorflow as tf
import tensorflow_probability as tfp

from thermo.bnn.hmc import hmc_predict
from thermo.bnn.map import map_predict
from thermo.plots import plot_output
from thermo.rf import rf_predict


# %%
# About the data: https://kaggle.com/c/boston-housing
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()

X_train, y_train = X_train.astype("float32"), y_train.astype("float32")
X_test, y_test = X_test.astype("float32"), y_test.astype("float32")


# %%
rf_y_pred, rf_y_std, rf_model = rf_predict(X_train, y_train, X_test)


# %%
plot_output(y_test, rf_y_pred, rf_y_std, title="RF")


# %%
print(f"RF MAE: {abs(rf_y_pred - y_test).mean():.3f}")


# %%
weight_prior = tfp.distributions.Normal(0, 0.2)
bias_prior = tfp.distributions.Normal(0, 0.2)
map_y_pred, map_y_var, map_log_probs, map_final_state = map_predict(
    weight_prior, bias_prior, X_train, y_train, X_test, y_test
)


# %%
print(f"RF MAE: {abs(map_y_pred - y_test).mean():.3f}")


# %%
plot_output(y_test, map_y_pred, map_y_var**0.5)


# %%
hmc_y_pred, hmc_y_var, _ = hmc_predict(
    weight_prior, bias_prior, map_final_state, X_train, y_train, X_test, y_test
)


# %%
print(f"RF MAE: {abs(hmc_y_pred - y_test).mean():.3f}")


# %%
plot_output(y_test, hmc_y_pred, hmc_y_var**0.5, title="HMC")
