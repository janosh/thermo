import os

import numpy as np
import pandas as pd
from tqdm import tqdm

# absolute path to the project's root directory
ROOT = os.getcwd().split("/src", 1)[0]


def pd2np(*args):
    return [a.values if isinstance(a, (pd.DataFrame, pd.Series)) else a for a in args]


def predict_multiple_labels(pred_func, X_train, y_train, X_test, y_test=None):
    """Train and run multiple models in sequence on a list of labels (e.g. rho, Seebeck,
    kappa, zT) passed as columns in the dataframe y_train. *args can contain
    additional arguments passed to the predictor function such as y_test for
    performance monitoring, e.g. calculating the log probability of weights in a BNN.
    """
    # Save column names and original dataframe index before converting to arrays.
    n_labels = len(y_train.columns)
    col_names, idx = y_train.columns, X_test.index

    X_train, y_train, X_test, y_test = pd2np(X_train, y_train, X_test, y_test)

    if not callable(pred_func):
        # If pred_func is not a function, it must be a list of functions,
        # one for each label.
        assert (
            len(pred_func) == n_labels
        ), f"len(pred_func) == {len(pred_func)} != len(y_train.columns) == {n_labels}"
        assert all(callable(fn) for fn in pred_func), "Received non-callable pred_func"
    else:
        pred_func = [pred_func] * n_labels

    # Calculate predictions (where all the work happens).
    iters = zip(pred_func, y_train.T, y_test.T if y_test else [None] * n_labels)
    results = [fn(X_train, y_tr, X_test, y_te) for fn, y_tr, y_te in iters]

    processed = [
        # convert lists and arrays to dataframes, restoring former label names and index
        pd.DataFrame(np.array(x).T, columns=col_names, index=idx)
        if isinstance(x[0], np.ndarray)
        # convert single-entry results (e.g. trained models) to dicts named by label
        else dict(zip(col_names, x))
        # transpose results so first dim is different result types (y_pred, y_var, etc.)
        # where before first dim was different labels
        for x in zip(*results)
    ]

    return processed


def sequence_to_df(dfs, swap_index_levels=False):
    # Adapted from https://stackoverflow.com/a/57338412.
    df = pd.concat(dfs)
    df = df.set_index(df.groupby(level=0).cumcount(), append=True).unstack(0)
    if swap_index_levels:
        df = df.swaplevel(0, 1, axis=1).sort_index(axis=1, ascending=[True, False])
    return df


def cross_val_predict(splitter, features, labels, predict_fn):
    results = []
    desc = f"{splitter.n_splits}-fold CV"

    for train_idx, test_idx in tqdm(splitter.split(features), desc=desc):

        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        output = predict_multiple_labels(
            predict_fn, X_train, y_train.T, X_test, y_test.T
        )
        results.append(output)

    return [
        pd.concat(x) if isinstance(x[0], pd.DataFrame) else x for x in zip(*results)
    ]
