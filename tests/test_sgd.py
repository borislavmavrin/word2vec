import numpy as np
from sgd import sgd


def test_sgd():
    """Test SGD algorithm."""
    def quad(x):
        return (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    t1, _ = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:", t1)
    assert abs(t1) <= 1e-6

    t2, _ = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:", t2)
    assert abs(t2) <= 1e-6

    t3, _ = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:", t3)
    assert abs(t3) <= 1e-6

    print("-" * 40)
    print("ALL TESTS PASSED")
    print("-" * 40)
