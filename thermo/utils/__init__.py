"""Project-wide utilities, paths and cross-validation helpers."""

from collections.abc import Callable
from os.path import dirname
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from tqdm import tqdm


# absolute path to the project's root directory
ROOT = dirname(dirname(dirname(__file__)))


def pd2np(*args: Any) -> list:
    """Convert pandas dataframes and series to numpy arrays."""
    return [
        a.to_numpy() if isinstance(a, (pd.DataFrame, pd.Series)) else a for a in args
    ]


def predict_multiple_targets(pred_func, x_train, y_train, x_test, y_test=None) -> list:
    """Train and run multiple models in sequence on a list of targets (e.g. rho,
    Seebeck, kappa, zT) passed as columns in the dataframe y_train. *args can contain
    additional arguments passed to the predictor function such as y_test for
    performance monitoring, e.g. calculating the log probability of weights in a BNN.
    """
    # Save column names and original dataframe index before converting to arrays.
    n_targets = len(y_train.columns)
    col_names, idx = y_train.columns, x_test.index

    x_train, y_train, x_test, y_test = pd2np(x_train, y_train, x_test, y_test)

    if not callable(pred_func):
        # If pred_func is not a function, it must be a list of functions,
        # one for each label.
        assert len(pred_func) == n_targets, f"{len(pred_func)=} != {n_targets=}"
        assert all(callable(fn) for fn in pred_func), "Received non-callable pred_func"
    else:
        pred_func = [pred_func] * n_targets

    # Calculate predictions (where all the work happens).
    iters = zip(
        pred_func,
        y_train.T,
        [None] * n_targets if y_test is None else y_test.T,
        strict=True,
    )
    results = [fn(x_train, y_tr, x_test, y_te) for fn, y_tr, y_te in iters]

    return [
        # convert lists and arrays to dataframes, restoring former label names and index
        pd.DataFrame(np.array(x).T, columns=col_names, index=idx)
        if isinstance(x[0], np.ndarray)
        # convert single-entry results (e.g. trained models) to dicts named by label
        else dict(zip(col_names, x, strict=True))
        # transpose results so first dim is different result types (y_pred, y_var, etc.)
        # where before first dim was different targets
        for x in zip(*results, strict=True)
    ]


def sequence_to_df(
    dfs: list[pd.DataFrame], *, swap_index_levels: bool = False
) -> pd.DataFrame:
    """Concatenate a list of dataframes and unstack the index."""
    # Adapted from https://stackoverflow.com/a/57338412.
    df_joined = pd.concat(dfs)
    df_joined = df_joined.set_index(  # noqa: PD010  # reshape, not aggregation
        df_joined.groupby(level=0).cumcount(), append=True
    ).unstack(0)
    assert isinstance(df_joined, pd.DataFrame)
    if swap_index_levels:
        df_joined = df_joined.swaplevel(0, 1, axis=1).sort_index(
            axis=1, ascending=[True, False]
        )
    return df_joined


def cross_val_predict(
    splitter: BaseCrossValidator,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    predict_fn: Callable | list[Callable],
) -> list:
    """Cross-validate a predictor function."""
    results = []
    n_splits = getattr(splitter, "n_splits", splitter.get_n_splits(features))
    for train_idx, test_idx in tqdm(
        splitter.split(features), desc=f"{n_splits}-fold CV"
    ):
        x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]

        output = predict_multiple_targets(predict_fn, x_train, y_train, x_test, y_test)
        results.append(output)

    return [
        pd.concat(x).sort_index() if isinstance(x[0], pd.DataFrame) else x
        for x in zip(*results, strict=True)
    ]
