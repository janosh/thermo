import matplotlib.pyplot as plt
from mlmatrics import err_decay, residual_hist, scatter_with_err_bar

from .uncertainty import ci_err_decay, mse_boxes, nm_to_mn_cols


def plot_output(y_test, y_pred, y_std=None, **kwargs):
    """Convenience function for generating multiple plots in one go for
    analyzing a model's accuracy and quality of uncertainty estimates.
    """
    fig1 = plt.gcf()
    scatter_with_err_bar(y_test, y_pred, yerr=y_std, **kwargs)
    plt.show()

    if y_std is None:
        return fig1

    fig2 = plt.gcf()
    err_decay(y_test, y_pred, y_std, **kwargs)
    plt.show()

    residual_hist(y_test, y_pred)
    plt.show()

    abs_err = abs(y_test - y_pred)
    fig3 = plt.gcf()
    scatter_with_err_bar(
        abs_err, y_std, xlabel="absolute error", ylabel="model uncertainty", **kwargs
    )
    plt.show()
    return fig1, fig2, fig3
