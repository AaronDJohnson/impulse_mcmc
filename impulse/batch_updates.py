import numpy as np


def update_mean(old_length, x_avg, x_new):
    """
    Batch update the mean
    Input:
        old_length (int): number of values in array up to now
        x_avg (array): average of the array columns up to now
        x_new (array): array of new values to take average of
    Return:
        x_avg (array): new means of array columns
    """
    n = old_length
    m = len(x_new)
    x_wgt = np.sum(x_new - x_avg, axis=0)
    return x_avg + x_wgt / (n + m)


def update_covariance(old_length, x_cov, x_avg, x_new):
    """
    Batch update covariance matrix
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