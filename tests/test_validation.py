import numpy as np
from culearn.utils.validation import as_tensor


def test_as_tensor_numpy():
    X = np.random.rand(10, 3)
    t = as_tensor(X)

    assert t.shape == (10, 3)
