"""If enough time, compare our implementation
with https://github.com/CitrineInformatics/lolo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor as SklRandomForestRegressor


if TYPE_CHECKING:
    from collections.abc import Sequence


Array = NDArray[np.float64]


class RandomForestRegressor(SklRandomForestRegressor):
    """Adapted from scikit-optimize.
    https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py.

    Uncertainty estimation: get_var() computes var(y|X_test) as described in sec. 4.3.2
    of https://arxiv.org/abs/1211.0906.
    """

    def __init__(self, *args: Sequence[Any], **kwargs: Any) -> None:
        self.params = {"args": args, **kwargs}
        super().__init__(*args, **kwargs)

    def get_params(self, _deep: bool = True) -> dict:
        """This method overrides the one inherited from sklearn.base.BaseEstimator
        which when trying to inspect instances of this class would throw a
        RuntimeError complaining that "scikit-learn estimators should always specify
        their parameters in the signature of their __init__ (no varargs).
        Constructor (self, *args, **kwargs) doesn't follow this convention.".
        sklearn enforces this to be able to read and set the parameter names
        in meta algorithms like pipeline and grid search which we don't need.
        """
        return self.params

    def predict(self, X_test: Array, uncertainty: str = "full") -> tuple[Array, Array]:
        """Predict y_pred and uncertainty y_var for X_test.

        Args:
            X_test (array-like, shape=(n_samples, n_features)): Input data.
            uncertainty (str): One of 'aleatoric', 'epistemic' or 'full'.

        Returns:
            2-tuple: y_pred and y_var
        """
        if self.criterion != "mse":
            raise ValueError(f"impurity must be 'mse', got {self.criterion}")

        y_pred = super().predict(X_test)
        y_var = self.get_var(X_test, y_pred, uncertainty)

        return y_pred, y_var

    def get_var(self, X_test: Array, y_pred: Array, uncertainty: str = "full") -> Array:
        """Uses law of total variance to compute var(Y|X_test) as
        Var(E[Y|Tree]) + E[Var(Y|Tree)]. The first term represents epistemic uncertainty
        and is captured by the variance over the means predicted by individual trees.
        This uncertainty increases the more different trees disagree. The second term
        represents aleatoric uncertainty and is captured by the average impurity across
        all leaf nodes that a given samples ends up in for different trees. This is
        only a proxy for aleatoric uncertainty but is motivated by the interpretation
        that a less pure node means the tree is less certain about how to draw decision
        boundaries for that sample.
        See https://arxiv.org/abs/1211.0906 (sec. 4.3.2) and
        https://arxiv.org/abs/1710.07283 (paragraphs following eq. 3).

        HINT: If including aleatoric uncertainty, consider increasing min_samples_leaf
        to say 10 (default: 1) to improve uncertainty quality. Unfortunately, this
        worsens predictive performance. TODO: Might be worth training two separate
        models when including aleatoric uncertainty, one solely to estimate y_var_aleat
        with min_samples_leaf > 1 and one solely for y_pred.

        Note: Another option for estimating confidence intervals is the prediction
        variability, i.e. how influential training set is for producing observed
        random forest predictions. Implemented in
        https://github.com/scikit-learn-contrib/forest-confidence-interval.
        Empirically (at least on our data), our method of obtaining y_var appears
        to be more accurate.

        Args:
            X_test (array-like, shape=(n_samples, n_features)): Input data.
            y_pred (array-like, shape=(n_samples,)): Prediction for each sample
                as returned by RFR.predict(X_test).

        Returns:
            array-like, shape=(n_samples,): variance of y_pred given X_test.
                Since self.criterion is set to "mse", var[i] ~= var(y | X_test[i]).
        """
        valid_uncerts = ["epistemic", "aleatoric", "full"]
        if uncertainty not in valid_uncerts:
            raise ValueError(
                f"uncertainty must be one of {valid_uncerts}, got {uncertainty}"
            )

        # trees is a list of fitted binary decision trees.
        trees = self.estimators_
        y_var_epist, y_var_aleat = np.zeros([2, len(X_test)])

        for tree in trees:
            # We use tree impurity as a proxy for aleatoric uncertainty.
            # Doesn't work well in experiments though.
            # leaf indices that each sample is predicted as.
            leaf_idx = tree.apply(X_test)
            # Grab the impurity of assigned leaves.
            y_var_tree = tree.tree_.impurity[leaf_idx]
            y_var_aleat += y_var_tree

            y_pred_tree = tree.predict(X_test)
            y_var_epist += y_pred_tree**2

        y_var_aleat /= len(trees)
        y_var_epist /= len(trees)
        y_var_epist -= y_pred**2

        if uncertainty == "aleatoric":
            return y_var_aleat

        if uncertainty == "epistemic":
            return y_var_epist

        return y_var_epist + y_var_aleat

    def get_corr(self, X_test: Array, alpha: float = 1e-12) -> Array:
        """Compute the matrix of Pearson correlation coefficients between
        random forest predictions for each sample in X_test.

        See https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html.

        Args:
            X_test (Array shape [n_samples, n_features]): Input data.
            alpha (float): additive

        Returns:
            Array shape [n_samples, n_samples]: Correlation matrix of samples in X_test.
        """
        # Each row in preds corresponds to a variable, in this case different samples
        # in X_test, while each column contains a series of observations corresponding
        # to predictions from different trees in the forest.
        preds = np.array([tree.predict(X_test) for tree in self.estimators_]).T

        # Ensure the correlation matrix is positive definite despite rounding errors.
        psd = np.eye(len(X_test)) * alpha

        return np.corrcoef(preds) + psd


def rf_predict(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array = None,
    uncertainty: str = "full",
    **kwargs,
) -> tuple[Array, Array, RandomForestRegressor]:
    """Fit a random forest to (X_train, y_train) and predict on X_test.

    y_test is unused and only serves to make the function callable by other
    functions that pass it in, like predict_multiple_targets().

    Args:
        X_train (Array): Training features
        y_train (Array): Training targets
        X_test (Array): Test features.
        y_test (Array, optional): Test targets (unused). Defaults to None.
        uncertainty (str, optional): Which uncertainty types to return. One of
            'aleatoric', 'epistemic' or 'full'. Defaults to "full".

    Returns:
        Tuple[Array, Array, RandomForestRegressor]: Predictions, their corresponding
            uncertainties and the forest that did the predicting.
    """
    forest = RandomForestRegressor(random_state=0, **kwargs)
    forest.fit(X_train, y_train)

    y_pred, y_var = forest.predict(X_test, uncertainty=uncertainty)
    return y_pred, y_var**0.5, forest
