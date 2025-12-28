import numpy as np

def euclidean_distance_matrix(X1, X2):
    """
    High-performance Euclidean distance using the Gram Trick.
    Returns a matrix of shape (N, M).
    """
    # X1: [N, F], X2: [M, F]
    x1_sq = np.sum(X1**2, axis=1, keepdims=True)  # [N, 1]
    x2_sq = np.sum(X2**2, axis=1, keepdims=True).T # [1, M]
    
    # dists = x1^2 - 2*x1*x2 + x2^2
    dists = x1_sq - 2.0 * (X1 @ X2.T) + x2_sq
    
    # Ensure no negative values due to float precision
    return np.sqrt(np.maximum(dists, 0))