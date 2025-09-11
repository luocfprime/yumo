import numpy as np

from yumo.geometry_utils import triangle_areas


def test_triangle_areas_two_triangles():
    # Two right triangles in the XY plane
    tri_vertices = np.array([
        [  # first triangle with area 0.5
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        [  # second triangle with area 2.0
            [0, 0, 0],
            [2, 0, 0],
            [0, 2, 0],
        ],
    ])
    areas = triangle_areas(tri_vertices)
    expected = np.array([0.5, 2.0])
    assert np.allclose(areas, expected)
