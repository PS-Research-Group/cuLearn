import torch


@torch.no_grad()
def pairwise_sqeuclidean(X: torch.Tensor, Y: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """
    Squared Euclidean distances between rows of X and rows of Y.

    Returns D with shape (n_x, n_y):
      D[i, j] = ||X[i] - Y[j]||^2

    Batches over X to reduce peak memory.
    """
    if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
        raise TypeError("X and Y must be torch tensors")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D tensors")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")
    if X.device != Y.device:
        raise ValueError("X and Y must be on the same device")
    if X.dtype != Y.dtype:
        raise ValueError("X and Y must have the same dtype")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    n = X.shape[0]
    m = Y.shape[0]

    # (m,)
    y_norm = (Y * Y).sum(dim=1)

    out = torch.empty((n, m), device=X.device, dtype=X.dtype)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]  # (b, d)

        x_norm = (xb * xb).sum(dim=1, keepdim=True)  # (b, 1)
        cross = xb @ Y.t()  # (b, m)

        out[start:end] = x_norm + y_norm.unsqueeze(0) - 2.0 * cross

    out.clamp_min_(0.0)
    return out


@torch.no_grad()
def topk_sqeuclidean_neighbors(
    X_query: torch.Tensor,
    X_train: torch.Tensor,
    k: int,
    batch_size: int = 2048,
):
    """
    For each row in X_query, find k nearest neighbors in X_train (squared Euclidean).

    Returns:
      distances: (n_query, k)
      indices:   (n_query, k) into X_train
    """
    if not isinstance(X_query, torch.Tensor) or not isinstance(X_train, torch.Tensor):
        raise TypeError("X_query and X_train must be torch tensors")
    if X_query.ndim != 2 or X_train.ndim != 2:
        raise ValueError("X_query and X_train must be 2D tensors")
    if X_query.shape[1] != X_train.shape[1]:
        raise ValueError("X_query and X_train must have the same number of features")
    if X_query.device != X_train.device:
        raise ValueError("X_query and X_train must be on the same device")
    if X_query.dtype != X_train.dtype:
        raise ValueError("X_query and X_train must have the same dtype")
    if k <= 0:
        raise ValueError("k must be > 0")
    if X_train.shape[0] < k:
        raise ValueError("k cannot be greater than number of training samples")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    n_query = X_query.shape[0]
    best_d = torch.empty((n_query, k), device=X_query.device, dtype=X_query.dtype)
    best_i = torch.empty((n_query, k), device=X_query.device, dtype=torch.long)

    for start in range(0, n_query, batch_size):
        end = min(start + batch_size, n_query)
        qb = X_query[start:end]

        D = pairwise_sqeuclidean(qb, X_train, batch_size=max(1, batch_size))  # (b, n_train)
        d, idx = torch.topk(D, k=k, dim=1, largest=False, sorted=True)

        best_d[start:end] = d
        best_i[start:end] = idx

    return best_d, best_i
