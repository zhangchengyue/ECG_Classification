import numpy as np
from ecg_classification.utils import cartesian

def test_cartesian_product():
    a = np.array([1, 2, 3])
    b = np.array([4, 5])

    expected = np.array([[1, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5]])
    assert np.all(cartesian(a, b) == expected)

    expected = np.array([[4, 1], [5, 1], [4, 2], [5, 2], [4, 3], [5, 3]])
    assert np.all(cartesian(b, a) == expected)