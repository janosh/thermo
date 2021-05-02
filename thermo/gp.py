from sklearn.gaussian_process import GaussianProcessRegressor, kernels


def gp_predict(X_train, y_train, X_test, y_test):
    kernel = kernels.Matern(nu=0.5)
    model = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, random_state=0
    )
    model.fit(X_train, y_train)
    y_pred, y_std = model.predict(X_test, return_std=True)
    return y_pred, y_std ** 2, model
