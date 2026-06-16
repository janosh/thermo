"""Gaussian process regression baseline."""

from sklearn.gaussian_process import GaussianProcessRegressor, kernels


def gp_predict(x_train, y_train, x_test, _y_test=None) -> tuple:
    """Predict with a Gaussian Process Regressor."""
    kernel = kernels.Matern(nu=0.5)
    model = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, random_state=0
    )
    model.fit(x_train, y_train)
    y_pred, y_std = model.predict(x_test, return_std=True)
    return y_pred, y_std**2, model
