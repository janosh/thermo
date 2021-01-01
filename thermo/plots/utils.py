from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


def add_text_box(text, loc="upper right"):
    # Add lw=0 to remove the black edge around the bounding box.
    # prop: keyword params passed to the Text instance inside AnchoredText.
    prop = {"bbox": {"lw": 0.5, "facecolor": "white"}}
    text_box = AnchoredText(text, borderpad=1, prop=prop, loc=loc, pad=0.2)
    plt.gca().add_artist(text_box)


def add_identity(axis, **line_kwargs):
    """Add a parity line (y = x) (aka identity) to the provided axis."""
    # zorder=0 ensures other plotted data displays on top of line
    default_kwargs = dict(alpha=0.5, zorder=0, linestyle="dashed", color="black")
    (identity,) = axis.plot([], [], **default_kwargs, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axis)
    # Update identity line when moving the plot in interactive
    # viewing mode to always extend to the plot's edges.
    axis.callbacks.connect("xlim_changed", callback)
    axis.callbacks.connect("ylim_changed", callback)
    return axis
