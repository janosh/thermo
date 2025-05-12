from functools import wraps
from time import perf_counter
from typing import Callable


def interruptible(orig_func: Callable = None, handler: Callable = None):
    """Allows to gracefully abort calls to the decorated function with ctrl + c."""

    def wrapper(func: Callable):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                handler() if handler else print(
                    f"\nDetected KeyboardInterrupt: Aborting call to {func.__name__}"
                )

        return wrapped_function

    if orig_func:
        return wrapper(orig_func)

    return wrapper


def timed(func: Callable) -> Callable:
    """Measures execution time of decorated functions."""

    @wraps(func)
    def timed_func(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {perf_counter() - start:.3g} sec")
        return result

    return timed_func


def squeeze(func: Callable) -> Callable:
    """Unpack single-entry lists from the decorated function's return value."""
    isiter = lambda x: isinstance(x, (list, tuple))

    @wraps(func)
    def squeezed_func(*args, **kwargs):
        result = func(*args, **kwargs)

        if isiter(result):
            result = [x[0] if isiter(x) and len(x) == 1 else x for x in result]
            if len(result) == 1:
                result = result[0]

        return result

    return squeezed_func
