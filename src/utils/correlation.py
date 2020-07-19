from random import shuffle

import numpy as np
import pandas as pd


def rand_obj_val_avr(corr_mat, n_select, repeats=50):
    """Simulates the random baseline for the discrete constrained optimization
    problem of finding the n_select least correlated candidates out of a pool of
    len(corr_mat) candidates as determined by the dataframe of pairwise correlations
    corr_mat, averaged over repeats repetitions.

    Args:
        corr_mat (dataframe): dataframe of pairwise correlations.
        n_select (int): number of candidates to be selected from the dataframe.
        repeats (int, optional): How many random draws to average over. Defaults to 50.

    Returns: float: The average objective value achieved by random choice.
    """
    avr, n_variables = 0, len(corr_mat)
    for _ in range(repeats):
        rand_seq = [1] * n_select + [0] * (n_variables - n_select)
        shuffle(rand_seq)
        avr += corr_mat.dot(rand_seq).dot(rand_seq)
    return avr / repeats


def expected_rand_obj_val(corr_mat, n_select):
    """Compute the expectation value for the discrete constrained optimization
    problem of finding the n_select least correlated candidates out of a pool of
    len(corr_mat) candidates pairwise correlated according to corr_mat.

    See https://math.stackexchange.com/questions/3315535 for mathematical details.
    """
    if corr_mat.to_numpy:
        corr_mat = corr_mat.to_numpy()

    try:
        zT_chol = pd.np.linalg.cholesky(corr_mat)
    except np.linalg.LinAlgError:
        # Handle correlation matrices that are only slightly non-positive definite
        # due to rounding errors.
        np.fill_diagonal(corr_mat, corr_mat.diagonal() + 1e-10)
        zT_chol = pd.np.linalg.cholesky(corr_mat)
        print(
            "Warning: a small offset (1e-10) was added to the diagonal "
            "of the correlation matrix to make it positive definite"
        )

    res, n_variables = 0, len(corr_mat)
    for j in range(n_variables):
        for i in range(j, n_variables):
            res += zT_chol[i][j] ** 2
            temp = 0
            for k in range(i + 1, n_variables):
                temp += zT_chol[i][j] * zT_chol[k][j]
            res += 2 * (n_select - 1) / (n_variables - 1) * temp

    return n_select / n_variables * res
