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
               sample_cov: np.ndarray,
               proposal_L: list|None = None
               ) -> tuple[list, list, list]:
    """
    Compute eigen-decomposition for parameter groups from covariance matrix.

    Uses ``np.linalg.eigh`` (symmetric eigendecomposition) on covariance
    submatrices corresponding to parameter groups.  Returns eigenvalues in
    descending order so that the convention matches the old SVD-based code.

    Also precomputes ``L = U * sqrt(S)`` for each group, which is the
    "square root" of the covariance needed by AM / SCAM proposals.

    Parameters
    ----------
    svd_U : list
        List to store eigenvectors for each group.
    svd_S : list
        List to store eigenvalues for each group.
    groups : list
        List of parameter index arrays, one for each group.
    sample_cov : np.ndarray
        Full covariance matrix, shape (n_params, n_params).
    proposal_L : list or None
        List to store precomputed ``U * sqrt(S)`` matrices.  If *None*,
        a new list is created.

    Returns
    -------
    tuple of (list, list, list)
        updated_U : list
            Updated list of eigenvectors for each group.
        updated_S : list
            Updated list of eigenvalues for each group.
        proposal_L : list
            Precomputed ``U * sqrt(S)`` matrices for each group.

    Examples
    --------
    >>> import numpy as np
    >>> groups = [[0, 1], [2, 3]]  # Two parameter groups
    >>> cov = np.eye(4)  # 4x4 covariance matrix
    >>> U_list = [None, None]
    >>> S_list = [None, None]
    >>> U_updated, S_updated, L = svd_groups(U_list, S_list, groups, cov)
    >>> # U_updated[0] contains 2x2 eigenvector matrix for first group
    >>> # S_updated[0] contains 2-element eigenvalue array for first group
    >>> # L[0] contains precomputed U * sqrt(S) for first group
    """
    if proposal_L is None:
        proposal_L = [None] * len(groups)
    for ct, group in enumerate(groups):
        covgroup = sample_cov[group][:, group]
        try:
            eigvals, eigvecs = np.linalg.eigh(covgroup)
        except np.linalg.LinAlgError:
            # eigh can fail on degenerate matrices early in sampling;
            # fall back to identity-like defaults so proposals still work.
            k = len(group)
            eigvals = np.ones(k)
            eigvecs = np.eye(k)
        # Reverse to descending order (matching old SVD convention)
        svd_S[ct] = eigvals[::-1]
        svd_U[ct] = eigvecs[:, ::-1]
        # Clamp negative eigenvalues (numerical noise) before sqrt
        sqrt_s = np.sqrt(np.maximum(svd_S[ct], 0.0))
        proposal_L[ct] = svd_U[ct] * sqrt_s[None, :]
    return svd_U, svd_S, proposal_L