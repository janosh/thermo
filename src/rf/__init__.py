from rf.forest import RandomForestRegressor


def rf_predict(X_train, y_train, X_test, y_test=None, **kwargs):
    """Fit an RF to (X_train, y_train) and predict on X_test.
    y_test is only there to match the signature of other predict
    functions and allow use in predict_multiple_labels().
    """
    uncertainty = kwargs.pop("uncertainty", None)

    forest = RandomForestRegressor(random_state=0, **kwargs)
    forest.fit(X_train, y_train)

    y_pred, y_var = forest.predict(X_test, uncertainty=uncertainty)
    return y_pred, y_var, forest
