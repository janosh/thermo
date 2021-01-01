from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from skopt.space import Categorical, Integer, Real

from thermo.hyper_opt import run_hyper_opt

space = (
    Categorical(
        ["Matern", "RBF", "RationalQuadratic", "ExpSineSquared"], name="kernel"
    ),
    Integer(0, 5, name="n_restarts_optimizer"),
    Real(1e-12, 1e-8, prior="log-uniform", name="alpha"),
)


def build_model_and_run(params, X_train, y_train, X_test):
    kernel = getattr(kernels, params.pop("kernel"))()
    model = GaussianProcessRegressor(kernel=kernel, **params, random_state=0)
    model.fit(X_train, y_train)
    y_pred, y_std = model.predict(X_test, return_std=True)
    y_std[y_std <= 0] = 1e-6
    return y_pred, y_std


run_hyper_opt(
    build_model_and_run, space=space, log_dir_model="gp/hyper_opt", label="resistivity"
)
