"""If enough time, compare our implementation
with https://github.com/CitrineInformatics/lolo.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR


class RandomForestRegressor(RFR):
    """Adapted from scikit-optimize.
    https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/learning/forest.py

    Uncertainty estimation: get_var() computes var(y|X_test) as described in sec. 4.3.2
    of https://arxiv.org/abs/1211.0906.
    """

    def __init__(self, *args, **kwargs):
        self.params = {"args": args, **kwargs}
        super().__init__(*args, **kwargs)

    def get_params(self, deep=True):
        """This method overrides the one inherited from sklearn.base.BaseEstimator
        which when trying to inspect instances of this class would throw a
        RuntimeError complaining that "scikit-learn estimators should always specify
        their parameters in the signature of their __init__ (no varargs).
        Constructor (self, *args, **kwargs) doesn't  follow this convention.".
        sklearn enforces this to be able to read and set the parameter names
        in meta algorithms like pipeline and grid search which we don't need.
        """
        return self.params

    def predict(self, X_test, uncertainty=None):
        """Predict y_pred and uncertainty y_var for X_test.

        Args:
            X_test (array-like, shape=(n_samples, n_features)): Input data.

        Returns:
            2- or 1-tuple: y_pred (and y_var)
        """
        assert self.criterion == "mse", f"impurity must be 'mse', got {self.criterion}"

        y_pred = super().predict(X_test)
        y_var = self.get_var(X_test, y_pred, uncertainty or "full")

        return y_pred, y_var

    def get_var(self, X_test, y_pred, uncertainty="full"):
        """Uses law of total variance to compute var(Y|X_test) as
        E[Var(Y|Tree)] + Var(E[Y|Tree]). The first term represents epistemic uncertainty
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
        valid_uncert = ["epistemic", "aleatoric", "full"]
        assert (
            uncertainty in valid_uncert
        ), f"uncertainty must be one of {valid_uncert}, got {uncertainty}"

        # trees is a list of fitted binary decision trees.
        trees = self.estimators_
        y_var_epist, y_var_aleat = np.zeros([2, len(X_test)])

        for tree in trees:
            # We use tree impurity as a proxy for aleatoric uncertainty.
            # Doesn't work well in experiments though.
            # leaf indices that each sample is predicted as.
            leaf_idx = tree.apply(X_test)
            # Grab the impurity of assigned leafs.
            y_var_tree = tree.tree_.impurity[leaf_idx]
            y_var_aleat += y_var_tree

            y_pred_tree = tree.predict(X_test)
            y_var_epist += y_pred_tree ** 2

        y_var_aleat /= len(trees)
        y_var_epist /= len(trees)
        y_var_epist -= y_pred ** 2

        if uncertainty == "aleatoric":
            return y_var_aleat

        if uncertainty == "epistemic":
            return y_var_epist

        y_var = y_var_epist + y_var_aleat
        return y_var

    def get_corr(self, X_test, alpha=1e-12):
        """Compute the matrix of Pearson correlation coefficients between
        random forest predictions for each sample in X_test. Each entry in the
        correlation matrix is the covariance between those random variables
        divided by their respective standard deviations. See
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html.

        Args:
            X_test (array, shape=(n_samples, n_features)): Input data.

        Returns:
            array, shape=(n_samples, n_samples): Correlation coefficients between
            samples in X_test.
        """
        # Each row in preds corresponds to a variable, in this case different samples
        # in X_test, while each column contains a series of observations corresponding
        # to predictions from different trees in the forest.
        preds = np.array([tree.predict(X_test) for tree in self.estimators_]).T

        # Ensure the correlation matrix is positive definite despite rounding errors.
        psd = np.eye(len(X_test)) * alpha

        return np.corrcoef(preds) + psd
