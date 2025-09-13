import numpy as np
import pytest

from yumo.geometry_utils import _tree_cache, query_scalar_field


def test_query_scalar_field_basic():
    # Create simple 3D points with scalar values
    data_points = np.array([
        [0.0, 0.0, 0.0, 1.0],  # value 1
        [1.0, 0.0, 0.0, 2.0],  # value 2
        [0.0, 1.0, 0.0, 3.0],  # value 3
    ])

    # Query exactly these points
    query_points = np.array([
        [0.0, 0.0, 0.0],  # expect 1
        [1.0, 0.0, 0.0],  # expect 2
        [0.0, 1.0, 0.0],  # expect 3
    ])

    result = query_scalar_field(query_points, data_points, cache=False)

    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result, expected)


def test_query_scalar_field_nearest_neighbor():
    data_points = np.array([
        [0.0, 0.0, 0.0, 10.0],
        [5.0, 0.0, 0.0, 20.0],
    ])

    # A point closer to the first data point
    query_points = np.array([
        [1.0, 0.0, 0.0],  # nearer to [0,0,0]
    ])

    result = query_scalar_field(query_points, data_points, cache=False)

    assert result.shape == (1,)
    assert result[0] == pytest.approx(10.0)


def test_query_scalar_field_with_cache():
    _tree_cache.clear()  # ensure clean state

    data_points = np.array([
        [0.0, 0.0, 0.0, 42.0],
        [1.0, 1.0, 1.0, 99.0],
    ])

    query_points = np.array([[0.0, 0.0, 0.0]])

    # Call twice with cache enabled
    result1 = query_scalar_field(query_points, data_points, cache=True)
    result2 = query_scalar_field(query_points, data_points, cache=True)

    # Should return cached result (same output)
    assert result1[0] == 42.0
    assert np.all(result1 == result2)

    # Tree should be stored in cache
    assert len(_tree_cache) == 1


def test_query_scalar_field_without_cache():
    _tree_cache.clear()

    data_points = np.array([
        [0.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 6.0],
    ])

    query_points = np.array([[0.0, 0.0, 0.0]])

    result = query_scalar_field(query_points, data_points, cache=False)
    assert result.shape == (1,)
    assert result[0] == 5.0

    # Without cache, cache dict should remain empty
    assert len(_tree_cache) == 0
