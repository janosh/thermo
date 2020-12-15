from skopt.space import Integer

from rf.forest import RandomForestRegressor
from rf.hyper_opt import run_hyper_opt

space = (
    Integer(1, 100, name="n_estimators"),
    # Categorical([None, 1, 10, 100, 1000], name="max_depth"),
    # Categorical([None, 1, 10, 100, 1000], name="max_leaf_nodes"),
    # Categorical(["mse", "mae"], name="criterion"),
    # Integer(1, 100, name="min_samples_split"),
)


def build_model_and_run(params, X_train, y_train, X_test):
    model = RandomForestRegressor(**params, random_state=0)
    model.fit(X_train, y_train)
    y_pred, y_var = model.predict(X_test)
    # y_var[y_var <= 0] = 1e-6
    return y_pred, y_var


run_hyper_opt(
    build_model_and_run,
    space=space,
    log_dir_model="rf/hyper_opt",
    label="resistivity",
)
