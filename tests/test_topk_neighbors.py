import torch
from culearn.metrics import topk_sqeuclidean_neighbors


def test_topk_neighbors_matches_naive_small():
    torch.manual_seed(0)
    X_train = torch.randn(20, 4)
    X_query = torch.randn(6, 4)

    d, idx = topk_sqeuclidean_neighbors(X_query, X_train, k=3, batch_size=2)

    D_ref = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(dim=2)
    d_ref, idx_ref = torch.topk(D_ref, k=3, dim=1, largest=False, sorted=True)

    assert d.shape == (6, 3)
    assert idx.shape == (6, 3)
    assert torch.allclose(d, d_ref, atol=1e-6, rtol=1e-6)
    assert torch.equal(idx, idx_ref)
