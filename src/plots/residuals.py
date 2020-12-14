import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


def residual(y_true, y_pred):
    fig = plt.figure(figsize=(8, 8))

    y_err = y_pred - y_true

    xmin = np.min(y_true) * 0.9
    xmax = np.max(y_true) / 0.9

    plt.plot(y_true, y_err, "o", alpha=0.5, label=None, mew=1.2, ms=5.2)
    plt.plot([xmin, xmax], [0, 0], "k--", alpha=0.5, label="ideal")

    plt.ylabel("Residual error (Units)")
    plt.xlabel("Actual value (Units)")
    plt.legend(loc="lower right")
    return fig


def residual_hist(y_true, y_pred):
    fig = plt.figure(figsize=(8, 8))

    y_err = y_pred - y_true
    plt.hist(y_err, bins=35, density=True, edgecolor="black")

    kde_true = gaussian_kde(y_err)  # kernel density estimation
    x_range = np.linspace(min(y_err), max(y_err), 100)

    plt.plot(x_range, kde_true(x_range), lw=2, color="red", label="kde")

    plt.xlabel("Residual error (Units)")
    plt.legend(loc=2, framealpha=0.5, handlelength=1)
    plt.show()
    return fig
