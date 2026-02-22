import torch

from culearn.utils.validation import as_tensor
from culearn.metrics import topk_sqeuclidean_neighbors


class KNNClassifier:
    """
    K-Nearest Neighbors classifier (PyTorch backend).

    Current support:
      - metric: squared Euclidean (via topk_sqeuclidean_neighbors)
      - weights: "uniform" or "distance"
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",   # "uniform" | "distance"
        batch_size: int = 2048,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be > 0")
        if weights not in ("uniform", "distance"):
            raise ValueError('weights must be "uniform" or "distance"')
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype = dtype

        # learned in fit
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self._class_to_index_ = None

    def fit(self, X, y):
        X = as_tensor(X, device=self.device, dtype=self.dtype)

        if isinstance(y, torch.Tensor):
            y_t = y
        else:
            y_t = torch.as_tensor(y)

        if y_t.ndim != 1:
            y_t = y_t.view(-1)

        if X.shape[0] != y_t.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        # Move labels to same device; keep labels as long for indexing
        y_t = y_t.to(device=X.device)

        # Build class mapping (supports non-0..C-1 labels)
        classes = torch.unique(y_t)
        classes, _ = torch.sort(classes)
        class_to_index = {int(c.item()): i for i, c in enumerate(classes)}

        self.X_train_ = X
        self.y_train_ = y_t
        self.classes_ = classes
        self._class_to_index_ = class_to_index

        return self

    @torch.no_grad()
    def kneighbors(self, X, return_distance: bool = True):
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before kneighbors().")

        Xq = as_tensor(X, device=self.X_train_.device, dtype=self.X_train_.dtype)

        k = self.n_neighbors
        if self.X_train_.shape[0] < k:
            raise ValueError("n_neighbors cannot be greater than number of training samples")

        d2, idx = topk_sqeuclidean_neighbors(
            X_query=Xq,
            X_train=self.X_train_,
            k=k,
            batch_size=self.batch_size,
        )
        if return_distance:
            return d2, idx
        return idx

    @torch.no_grad()
    def predict_proba(self, X):
        if self.X_train_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        d2, idx = self.kneighbors(X, return_distance=True)
        y_neighbors = self.y_train_[idx]  # (n, k)

        # Map neighbor labels to class indices
        # We do this via a lookup vector for speed
        classes = self.classes_
        num_classes = classes.numel()

        # Build lookup table for integer labels (safe for typical class labels)
        # If labels are huge/sparse, this could be changed later.
        min_c = int(classes.min().item())
        max_c = int(classes.max().item())
        lut_size = max_c - min_c + 1
        lut = torch.full((lut_size,), -1, device=classes.device, dtype=torch.long)
        for i, c in enumerate(classes):
            lut[int(c.item()) - min_c] = i

        y_idx = lut[(y_neighbors.to(torch.long) - min_c).clamp(0, lut_size - 1)]
        if (y_idx < 0).any():
            raise ValueError("Found labels not present in fitted classes_ mapping.")

        n = y_idx.shape[0]
        probs = torch.zeros((n, num_classes), device=self.X_train_.device, dtype=self.X_train_.dtype)

        if self.weights == "uniform":
            # Count votes per class
            ones = torch.ones_like(d2, dtype=self.X_train_.dtype)
            probs.scatter_add_(1, y_idx, ones)
        else:
            # distance weights: weight = 1 / (sqrt(d2) + eps)
            # Use sqrt to match common KNN distance weighting semantics
            eps = 1e-12
            w = 1.0 / (torch.sqrt(d2) + eps)
            probs.scatter_add_(1, y_idx, w)

        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return probs

    @torch.no_grad()
    def predict(self, X):
        probs = self.predict_proba(X)
        pred_class_idx = torch.argmax(probs, dim=1)
        return self.classes_[pred_class_idx]

    @torch.no_grad()
    def score(self, X, y):
        y_pred = self.predict(X)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)
        y = y.view(-1).to(device=y_pred.device)
        return (y_pred == y).float().mean().item()
