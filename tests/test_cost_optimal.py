import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from rul_fusion.cost_optimal import (apply_cost_optimal, cost_optimal_shift,
                                    expected_cost)


def test_symmetric_cost_keeps_median():
    """With a symmetric cost the optimal estimate is the median (zero shift)."""
    q10 = np.array([10.0, 40.0])
    q50 = np.array([20.0, 50.0])
    q90 = np.array([30.0, 60.0])

    def symmetric(err):
        return np.abs(err)

    import rul_fusion.cost_optimal as co
    original = co._cost
    co._cost = symmetric
    try:
        shift = cost_optimal_shift(q10, q50, q90, search=np.arange(-6.0, 6.01, 0.5))
    finally:
        co._cost = original
    assert np.allclose(shift, 0.0)


def test_asymmetric_cost_shifts_toward_the_cheap_side():
    """Under the NASA cost, under-prediction is cheaper, so the estimate rises."""
    q10 = np.array([10.0])
    q50 = np.array([20.0])
    q90 = np.array([30.0])
    _, shift = apply_cost_optimal(q10, q50, q90)
    assert shift[0] > 0.0


def test_expected_cost_is_minimised_at_the_returned_shift():
    q10 = np.array([10.0, 100.0])
    q50 = np.array([20.0, 110.0])
    q90 = np.array([30.0, 120.0])
    shift = cost_optimal_shift(q10, q50, q90)
    at_best = expected_cost(q50 + shift, q10, q50, q90)
    for alternative in (-4.0, -2.0, 2.0, 4.0):
        assert np.all(at_best <= expected_cost(q50 + alternative, q10, q50, q90) + 1e-9)


def test_shift_is_finite_and_bounded():
    q10 = np.array([0.0, 50.0])
    q50 = np.array([125.0, 60.0])
    q90 = np.array([125.0, 125.0])
    adjusted, shift = apply_cost_optimal(q10, q50, q90)
    assert np.isfinite(adjusted).all() and np.isfinite(shift).all()
    assert np.abs(shift).max() <= 8.0
