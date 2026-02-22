import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier as SKKNN

from culearn.neighbors import KNNClassifier


def test_knn_classifier_matches_sklearn_uniform():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)).astype(np.float32)

    # Make labels stable (avoid lots of ties)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int64)

    X_train, X_test = X[:100], X[100:]
    y_train, y_test = y[:100], y[100:]

    k = 5

    sk = SKKNN(n_neighbors=k, weights="uniform", metric="euclidean")
    sk.fit(X_train, y_train)
    sk_pred = sk.predict(X_test)

    cl = KNNClassifier(n_neighbors=k, weights="uniform", batch_size=32, device=None)
    cl.fit(torch.from_numpy(X_train), torch.from_numpy(y_train))
    cl_pred = cl.predict(torch.from_numpy(X_test)).cpu().numpy()

    assert np.array_equal(sk_pred, cl_pred)


def test_knn_classifier_predict_proba_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    model = KNNClassifier(n_neighbors=3, weights="uniform", batch_size=16)
    model.fit(X, y)

    proba = model.predict_proba(X[:7])
    assert proba.shape == (7, 2)
    # rows sum to 1
    row_sums = proba.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6)
