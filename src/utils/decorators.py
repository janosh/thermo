import functools
import os
from time import time

import IPython
from matplotlib import pyplot as plt


def with_attr(key, val):
    def wrapper(func):
        setattr(func, key, val)
        return func

    return wrapper


def timed(func):
    """Measures execution time of decorated functions."""

    @functools.wraps(func)
    def timed_func(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {round(time() - start, 1)} sec")
        return result

    return timed_func


def run_once(func):
    """ensures the decorated function runs at most once in a session"""

    @functools.wraps(func)
    def once_running_func(*args, **kwargs):
        if once_running_func.has_run is False:
            once_running_func.has_run = True
            return func(*args, **kwargs)

    once_running_func.has_run = False
    return once_running_func


def squeeze(func):
    """unpacks single-entry lists from the decorated function's return value"""

    isiter = lambda x: isinstance(x, (list, tuple))

    @functools.wraps(func)
    def squeezed_func(*args, **kwargs):
        result = func(*args, **kwargs)

        if isiter(result):
            result = [x[0] if isiter(x) and len(x) == 1 else x for x in result]
            if len(result) == 1:
                result = result[0]

        return result

    return squeezed_func


def handle_plot(func):
    """Decorator for plotting functions. In a regular script, it saves the plot to a path
    passed as `path` kwarg to the decorated function. If used in interactive Python,
    it just displays the output with plt.show() unless the env variable save_to_disk
    is set to "True" in which case it also saves the plot to a file at path.
    """

    def savefig(*args, show=False, **kwargs):
        path = kwargs.pop("path")
        result = func(*args, **kwargs)
        # plt.savefig must come before plt.show, else the plot will be empty.
        plt.savefig(path, bbox_inches="tight", transparent=True)
        if show:
            plt.show()
        plt.close()
        return result

    @functools.wraps(func)
    def saved_plot(*args, **kwargs):
        if not in_ipython() and "path" in kwargs:
            return savefig(*args, **kwargs)
        elif in_ipython():
            if os.getenv("save_to_disk") == "True":
                result = savefig(*args, show=True, **kwargs)
            else:
                kwargs.pop("path", None)
                result = func(*args, **kwargs)
                plt.show()
            return result
        elif not in_ipython() and "path" not in kwargs:
            print(
                f"Warning: Not running in interactive notebook and not passing "
                f"a path kwarg to decorated plotting function. Expected at "
                f"least one to be true. Skipping function {func.__name__}."
            )

    return saved_plot


def in_ipython():
    """Returns a boolean indicating whether Python is running in interactive mode.
    To simulate iPython behavior when using a debugger, simply return True.
    """
    return IPython.get_ipython() is not None
