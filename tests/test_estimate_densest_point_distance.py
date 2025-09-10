import numpy as np
import pytest

from yumo.utils import estimate_densest_point_distance


def test_linear_points_exact_distance():
    """
    Tests the function with a simple, predictable set of 1D points.
    """
    points = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    expected_distance = 1.0
    estimated_distance = estimate_densest_point_distance(points, k=5)
    assert estimated_distance == pytest.approx(expected_distance)


def test_2d_grid_points():
    """
    Tests the function with a predictable 2D point set (corners of a square).
    """
    points = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0], [20.0, 20.0], [10.0, 20.0], [20.0, 10.0]])
    expected_distance = 10.0
    estimated_distance = estimate_densest_point_distance(points, k=4)
    assert estimated_distance == pytest.approx(expected_distance)


def test_2d_grid_points_outliers():
    """
    Tests the function with a predictable 2D point set with outliers. See if it filters out the outliers.
    """
    data = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 0.0], [10.0, 10.0], [20.0, 20.0], [10.0, 20.0], [20.0, 10.0]])
    noises = np.array([[50.0, 100.0], [89.0, 78.0]])

    points = np.concatenate((data, noises))
    expected_distance = 10.0
    estimated_distance = estimate_densest_point_distance(points, k=4)
    assert estimated_distance == pytest.approx(expected_distance)


def test_identical_points_zero_distance():
    """
    Tests the function with a special case where all points are identical.
    """
    points = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    expected_distance = 0.0
    estimated_distance = estimate_densest_point_distance(points, k=4)
    assert estimated_distance == pytest.approx(expected_distance)


def test_sampling_logic_is_approximate():
    """
    Tests the random sampling logic of the function.

    This test verifies that when k is less than the total number of points,
    the function produces an approximate result. It also shows that setting a
    random seed makes the "random" sampling deterministic for test reproducibility.
    """
    # 1. Set a seed for reproducible random numbers
    np.random.seed(42)

    # 2. Create a point cloud with more points than the sampling size
    points = np.random.rand(1000, 3)

    # 3. Calculate the "exact" average distance by using all points (k=100)
    exact_distance = estimate_densest_point_distance(points, k=1000)

    # 4. Reset the seed to get a consistent sample for the first test run
    np.random.seed(42)
    # Calculate the estimated distance by sampling only 20 points
    sampled_distance_run1 = estimate_densest_point_distance(points, k=200)

    # 5. Calculate another sampled distance without resetting the seed
    # This will produce a different sample and thus a different result
    sampled_distance_run2 = estimate_densest_point_distance(points, k=200)

    # --- Assertions ---

    # The result from the first sampled run should be the same if the seed is reset
    np.random.seed(42)
    assert sampled_distance_run1 == estimate_densest_point_distance(points, k=200)

    # The two different random samples should produce different results
    assert sampled_distance_run1 != sampled_distance_run2, "Two consecutive random samples should not be identical."

    # The sampled result is an approximation, so it shouldn't (usually) be exact
    assert sampled_distance_run1 != exact_distance, (
        "Sampled result should not be exactly equal to the full calculation."
    )

    # The sampled result should be reasonably close to the exact result.
    # We allow for a 30% relative tolerance for this random test.
    assert exact_distance == pytest.approx(sampled_distance_run1, rel=0.3)
