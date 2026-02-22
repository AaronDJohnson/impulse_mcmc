import numpy as np

def update_mean(old_arr_length: int,
                old_arr_avg: np.ndarray,
                new_arr: np.ndarray,
                ) -> np.ndarray:
    """
    Batch update the running mean with new samples.

    Parameters
    ----------
    old_arr_length : int
        Number of samples seen so far.
    old_arr_avg : np.ndarray
        Current mean estimate.
    new_arr : np.ndarray
        New samples to incorporate, shape (n_new, ndim).

    Returns
    -------
    np.ndarray
        Updated mean estimate.
    """
    weight = np.sum(new_arr - old_arr_avg, axis=0)
    return old_arr_avg + weight / (old_arr_length + len(new_arr))

def update_covariance(old_arr_length: int,
                      old_arr_cov: np.ndarray,
                      old_arr_avg: np.ndarray,
                      new_arr: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch update the running sample covariance matrix.

    Parameters
    ----------
    old_arr_length : int
        Number of samples seen so far.
    old_arr_cov : np.ndarray
        Current covariance estimate, shape (ndim, ndim).
    old_arr_avg : np.ndarray
        Current mean estimate, shape (ndim,).
    new_arr : np.ndarray
        New samples to incorporate, shape (n_new, ndim).

    Returns
    -------
    new_mean : np.ndarray
        Updated mean estimate.
    new_cov : np.ndarray
        Updated covariance estimate.
    """
    n = old_arr_length
    m = len(new_arr)
    x_avgnew = update_mean(n, old_arr_avg, new_arr)
    y = new_arr - x_avgnew
    z = new_arr - old_arr_avg
    cov = ((n - 1) * old_arr_cov + np.dot(y.T, z)) / (n + m - 1)
    return x_avgnew, cov

def svd_groups(svd_U: list,
               svd_S: list,
               groups: list,
               sample_cov: np.ndarray
               ) -> tuple[list, list]:
    """
    Compute SVD decomposition for parameter groups from covariance matrix.

    Performs singular value decomposition on covariance submatrices
    corresponding to parameter groups. This enables efficient adaptive
    proposals by working in the principal component space.

    Parameters
    ----------
    svd_U : list
        List to store left singular vectors (eigenvectors) for each group.
    svd_S : list
        List to store singular values (square roots of eigenvalues) for each group.
    groups : list
        List of parameter index arrays, one for each group.
    sample_cov : np.ndarray
        Full covariance matrix, shape (n_params, n_params).

    Returns
    -------
    tuple of (list, list)
        updated_U : list
            Updated list of left singular vectors for each group.
        updated_S : list
            Updated list of singular values for each group.

    Examples
    --------
    >>> import numpy as np
    >>> groups = [[0, 1], [2, 3]]  # Two parameter groups
    >>> cov = np.eye(4)  # 4x4 covariance matrix
    >>> U_list = [None, None]
    >>> S_list = [None, None]
    >>> U_updated, S_updated = svd_groups(U_list, S_list, groups, cov)
    >>> # U_updated[0] contains 2x2 eigenvector matrix for first group
    >>> # S_updated[0] contains 2-element eigenvalue array for first group
    """
    for ct, group in enumerate(groups):
        covgroup = sample_cov[group][:, group]
        svd_U[ct], svd_S[ct], __ = np.linalg.svd(covgroup)
    return svd_U, svd_S