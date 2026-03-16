import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    Return None if input is invalid.
    """
    X = np.array(X)

    # Check if X is 2D
    if X.ndim != 2:
        return None

    n = X.shape[0]

    # Need at least 2 samples
    if n < 2:
        return None

    # Compute mean
    mu = np.mean(X, axis=0)

    # Center the data
    X_centered = X - mu

    # Covariance matrix
    cov = (X_centered.T @ X_centered) / (n - 1)

    return cov