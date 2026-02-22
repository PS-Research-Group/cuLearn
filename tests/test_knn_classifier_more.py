import numpy as np
import pytest
import torch
from sklearn.neighbors import KNeighborsClassifier as SKKNN

from culearn.neighbors import KNNClassifier


def test_knn_classifier_matches_sklearn_distance():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(140, 5)).astype(np.float32)
    y = (X[:, 0] - 0.2 * X[:, 1] + 0.1 > 0).astype(np.int64)

    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]

    k = 7

    sk = SKKNN(n_neighbors=k, weights="distance", metric="euclidean")
    sk.fit(X_train, y_train)
    sk_pred = sk.predict(X_test)

    cl = KNNClassifier(n_neighbors=k, weights="distance", batch_size=32)
    cl.fit(torch.from_numpy(X_train), torch.from_numpy(y_train))
    cl_pred = cl.predict(torch.from_numpy(X_test)).cpu().numpy()

    assert np.array_equal(sk_pred, cl_pred)


def test_knn_raises_if_k_gt_ntrain():
    X_train = torch.randn(5, 3)
    y_train = torch.tensor([0, 1, 0, 1, 0])

    model = KNNClassifier(n_neighbors=6)
    with pytest.raises(ValueError):
        model.fit(X_train, y_train)
