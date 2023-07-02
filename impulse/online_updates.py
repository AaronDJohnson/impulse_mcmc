import numpy as np
from loguru import logger


def update_mean(
    old_arr_length: int,
    old_arr_avg: float,
    new_arr: np.ndarray,
    ) -> float:
    """
    Batch update the mean
    Input:
        old_length (int): number of values in array up to now
        old_arr_avg (ndarray): average of the array columns up to now
        new_arr (ndarray): array of new values to take average of
    Return:
        new_avg (array): new means of array columns
    """
    weight = np.sum(new_arr - old_arr_avg, axis=0)
    return old_arr_avg + weight / (old_arr_length + len(new_arr))


# TODO(Aaron): fix variable names on the following two functions:
def update_covariance(old_length, x_cov, x_avg, x_new):
    """
    Batch update sample covariance matrix
    Input:
        old_length (int): number of values in array up to now
        x_cov (array): covariance of the array columns up to now
        x_avg (array): average of the array columns up to now
        x_new (array): new array to update with
    Return:
        x_cov (array): new covariance of array columns
    """
    n = old_length
    m = len(x_new)
    x_avgnew = update_mean(n, x_avg, x_new)
    y = x_new - x_avgnew
    z = x_new - x_avg
    cov = ((n - 1) * x_cov + np.dot(y.T, z)) / (n + m - 1)
    return x_avgnew, cov


def svd_groups(U, S, groups, cov):
    # do svd on parameter groups
    # TODO(Aaron): Speed this up using broadcasting/jit?
    for ct, group in enumerate(groups):
        covgroup = np.zeros((len(group), len(group)))
        for ii in range(len(group)):
            for jj in range(len(group)):
                covgroup[ii, jj] = cov[group[ii], group[jj]]
        U[ct], S[ct], __ = np.linalg.svd(covgroup)
    return U, S


# try this one:
# def svd_groups(U, S, groups, cov):
#     # do svd on parameter groups
#     for ct, group in enumerate(groups):
#         covgroup = cov[group][:, group]
#         U[ct], S[ct], __ = np.linalg.svd(covgroup)
#     return U, S