import torch
from culearn.metrics import pairwise_sqeuclidean


def test_pairwise_sqeuclidean_matches_naive():
    torch.manual_seed(0)
    X = torch.randn(17, 5)
    Y = torch.randn(9, 5)

    D = pairwise_sqeuclidean(X, Y, batch_size=7)
    D_ref = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(dim=2)

    assert D.shape == (17, 9)
    assert torch.allclose(D, D_ref, atol=1e-6, rtol=1e-6)
    assert torch.min(D).item() >= 0.0
