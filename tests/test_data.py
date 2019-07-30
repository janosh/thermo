import numpy as np

from data import dropna, normalize


def all_eq(arr1, arr2):
    return (arr1 == arr2).all()


def all_close(arr1, arr2):
    return np.isclose(arr1, arr2).all()


def test_dropna():
    # 1d arrays
    arr1 = np.array([np.nan, 0])
    arr2 = np.array([0, 1])
    arr1, arr2 = dropna(arr1, arr2)
    assert all_eq(arr1, np.array([0])), "dropna fails on first arg"
    assert all_eq(arr2, np.array([1])), "dropna fails on second arg"

    # 2d arrays
    arr1, arr2 = np.arange(20, dtype=float).reshape(2, 2, 5)
    arr1[0][0] = np.nan
    arr1, arr2 = dropna(arr1, arr2)
    assert all_eq(arr1, np.arange(5, 10))
    assert all_eq(arr2, np.arange(15, 20))


def test_normalize():
    n_rows, n_cols = 4, 5
    arr = np.random.rand(n_rows, n_cols)
    arr_norm, [arr_mean, arr_std] = normalize(arr)

    assert all_close(arr_norm.mean(0), np.zeros(n_cols)), "creates non-zero mean"
    assert all_close(arr_norm.std(0), np.ones(n_cols)), "creates non-unit std"

    assert all_close(arr.mean(0), arr_mean), "returns incorrect mean"
    assert all_close(arr.std(0), arr_std), "returns incorrect std"
