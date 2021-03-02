from typing import Tuple

from numpy import ndarray as Array

from thermo.rf.forest import RandomForestRegressor


def rf_predict(
    X_train: Array,
    y_train: Array,
    X_test: Array,
    y_test: Array = None,
    uncertainty: str = "full",
    **kwargs
) -> Tuple[Array, Array, RandomForestRegressor]:
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
    return y_pred, y_var ** 0.5, forest
