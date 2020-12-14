from .elements import hist_elemental_prevalence, ptable_elemental_prevalence
from .nn import log_probs, loss_history
from .residuals import residual, residual_hist
from .true_vs_pred import scatter_with_hist, true_vs_pred
from .uncertainty import abs_err_vs_std, ci_err_decay, err_decay, mse_boxes


def plot_output(y_test, y_pred, y_std=None, **kwargs):
    """Convenience function for generating multiple plots in one go for
    analyzing a model's accuracy and quality of uncertainty estimates.
    """
    fig1 = true_vs_pred(y_test, y_pred, y_std=y_std, **kwargs)
    # fig4 = residual(y_test, y_pred)
    # fig5 = residual_hist(y_test, y_pred)
    if y_std is None:
        return

    fig2 = err_decay(y_test, y_pred, y_std, **kwargs)

    abs_err = abs(y_test - y_pred)
    fig3 = abs_err_vs_std(abs_err, y_std, **kwargs)
    return fig1, fig2, fig3


def show_bar_values(ax=None, offset=15):
    """Annotate histograms with a label indicating the height/count of each bar.

    Args:
        ax (matplotlib.axes.Axes): The axes to annotate.
        offset (int): The distance between the labels and the bars.
    """
    for rect in ax.patches:
        y_val = rect.get_height()
        x_val = rect.get_x() + rect.get_width() / 2

        # place label at end of the bar and center horizontally
        ax.annotate(y_val, (x_val, y_val + offset), ha="center")
        # ensure enough vertical space to display label above highest bar
        ax.margins(y=0.1)
