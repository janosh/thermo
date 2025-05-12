from os.path import dirname

import numpy as np
import pandas as pd
from tqdm import tqdm


# absolute path to the project's root directory
ROOT = dirname(dirname(dirname(__file__)))


def pd2np(*args):
    return [a.values if isinstance(a, (pd.DataFrame, pd.Series)) else a for a in args]


def predict_multiple_targets(pred_func, X_train, y_train, X_test, y_test=None):
    """Train and run multiple models in sequence on a list of targets (e.g. rho,
    Seebeck, kappa, zT) passed as columns in the dataframe y_train. *args can contain
    additional arguments passed to the predictor function such as y_test for
    performance monitoring, e.g. calculating the log probability of weights in a BNN.
    """
    # Save column names and original dataframe index before converting to arrays.
    n_targets = len(y_train.columns)
    col_names, idx = y_train.columns, X_test.index

    X_train, y_train, X_test, y_test = pd2np(X_train, y_train, X_test, y_test)

    if not callable(pred_func):
        # If pred_func is not a function, it must be a list of functions,
        # one for each label.
        assert len(pred_func) == n_targets, f"{len(pred_func)=} != {n_targets=}"
        assert all(callable(fn) for fn in pred_func), "Received non-callable pred_func"
    else:
        pred_func = [pred_func] * n_targets

    # Calculate predictions (where all the work happens).
    iters = zip(
        pred_func, y_train.T, [None] * n_targets if y_test is None else y_test.T
    )
    results = [fn(X_train, y_tr, X_test, y_te) for fn, y_tr, y_te in iters]

    return [
        # convert lists and arrays to dataframes, restoring former label names and index
        pd.DataFrame(np.array(x).T, columns=col_names, index=idx)
        if isinstance(x[0], np.ndarray)
        # convert single-entry results (e.g. trained models) to dicts named by label
        else dict(zip(col_names, x))
        # transpose results so first dim is different result types (y_pred, y_var, etc.)
        # where before first dim was different targets
        for x in zip(*results)
    ]


def sequence_to_df(dfs, swap_index_levels=False):
    # Adapted from https://stackoverflow.com/a/57338412.
    df_joined = pd.concat(dfs)
    df_joined = df_joined.set_index(
        df_joined.groupby(level=0).cumcount(), append=True
    ).unstack(0)
    if swap_index_levels:
        df_joined = df_joined.swaplevel(0, 1, axis=1).sort_index(
            axis=1, ascending=[True, False]
        )
    return df_joined


def cross_val_predict(splitter, features, targets, predict_fn):
    results = []
    for train_idx, test_idx in tqdm(
        splitter.split(features), desc=f"{splitter.n_splits}-fold CV"
    ):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]

        output = predict_multiple_targets(predict_fn, X_train, y_train, X_test, y_test)
        results.append(output)

    return [
        pd.concat(x).sort_index() if isinstance(x[0], pd.DataFrame) else x
        for x in zip(*results)
    ]
