from time import time

import numpy as np
import skopt
from sklearn.metrics import mean_squared_error as mse

from data import Data
from utils import ROOT
from utils.evaluate import plot_output


def run_hyper_opt(
    build_model_and_run,
    space,
    label,
    n_calls=10,
    minimizer="gp_minimize",
    callbacks=[],
):
    assert label in ["resistivity", "seebeck", "therm_cond"], "Invalid label"

    data = Data(label)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data.split((0.8, 0.1, 0.1))

    settings_msg = (
        f"Running gaussian process hyper optimization on label: {label} for {n_calls} "
        f"iterations with mse loss."
    )
    print(settings_msg)
    param_names = [x._name for x in space]

    def objective(params):
        params = dict(zip(param_names, params))
        y_pred, _ = build_model_and_run(params, X_train, y_train, X_val)
        loss = mse(y_val, y_pred)
        return loss

    log_cb = LoggerCallback(param_names, n_calls)
    cp_saver = skopt.callbacks.CheckpointSaver(
        ROOT + "/logs/hyperopt_checkpoint.pkl", store_objective=False
    )
    callbacks.extend((log_cb, cp_saver))
    hyperopt_res = getattr(skopt, minimizer)(
        objective, space, n_calls=n_calls, random_state=0, callback=callbacks
    )
    print(f"hyper optimization results: {hyperopt_res}")

    best_params = dict(zip(param_names, hyperopt_res.x))
    print(f"Test set performance with best hyperparameters {best_params}")

    # train model with best parameters on combined training + validation set
    y_pred, y_std = build_model_and_run(
        best_params,
        np.concatenate((X_train, X_val)),
        np.concatenate((y_train, y_val)),
        X_test,
    )
    plot_output(y_test, y_pred, y_std)


class LoggerCallback:
    def __init__(self, param_names, n_calls):
        self.time = time()
        self.iter_times = []
        self.param_names = param_names
        self.n_calls = n_calls

    def __call__(self, results_so_far):
        elapsed_time = round(time() - self.time, 4)
        self.iter_times.append(elapsed_time)
        results_so_far.iter_times = self.iter_times

        print(
            f"iteration {len(results_so_far.func_vals)}"
            "/{self.n_calls} took {elapsed_time} sec"
        )
        last_params = dict(zip(self.param_names, results_so_far.x_iters[-1]))
        print(f"hyperparameters: {last_params}")
        print(f"mse: {round(results_so_far.func_vals[-1], 4)}")
        print(f"current best: {round(results_so_far.fun, 4)}")
        self.time = time()
