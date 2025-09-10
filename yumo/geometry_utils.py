import logging

import numpy as np
import xatlas
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


def unwrap_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs UV unwrapping for a given 3D mesh using the xatlas library.

    This function takes a mesh defined by vertices and faces, generates a UV atlas,
    and returns the unwrapped mesh data along with the atlas properties. The UV
    atlas maps the 3D surface of the mesh onto a 2D plane, which is essential
    for applying textures.

    Args:
        vertices (np.ndarray): A NumPy array of shape (N, 3) representing the
            3D coordinates of the N mesh vertices.
        faces (np.ndarray): A NumPy array of shape (M, 3) representing the M
            triangular faces of the mesh. Each face is defined by three indices
            into the `vertices` array.

    Returns:
        Tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A tuple containing the following elements:
        - param_corner (np.ndarray): Shape (M * 3, 2). UV coordinates for each
          corner of each unwrapped face. Useful for some rendering APIs.
        - texture_height (int): The height of the generated texture atlas in pixels.
        - texture_width (int): The width of the generated texture atlas in pixels.
        - vmapping (np.ndarray): Shape (V,). An index array of length V (number of
          unwrapped vertices) that maps each unwrapped vertex back to its
          original vertex index in the input `vertices` array.
        - faces_unwrapped (np.ndarray): Shape (M, 3). The face index array for the
          new unwrapped mesh. The indices refer to the `vertices_unwrapped` and `uvs` arrays.
        - uvs (np.ndarray): Shape (V, 2). The UV coordinates for each of the V
          unwrapped vertices. The coordinates are normalized to the range [0, 1].
        - vertices_unwrapped (np.ndarray): Shape (V, 3). The 3D coordinates of the
          V unwrapped vertices. This is essentially `vertices[vmapping]`.
    """
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart_options = xatlas.ChartOptions()
    atlas.generate(chart_options=chart_options)
    vmapping, faces_unwrapped, uvs = atlas[0]  # [N], [M, 3], [N, 2]

    vertices_unwrapped = vertices[vmapping]

    param_corner = uvs[faces_unwrapped].reshape(-1, 2)

    texture_height, texture_width = atlas.height, atlas.width

    return param_corner, texture_height, texture_width, vmapping, faces_unwrapped, uvs, vertices_unwrapped


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    points_per_area: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized surface sampling on a triangular mesh.

    Args:
        vertices (np.ndarray): (N, 3). Vertex positions.
        faces (np.ndarray): (M, 3). Triangle vertex indices.
        points_per_area (float): Density (points per unit area).
        rng (np.random.Generator, optional): Random generator to use.
            If None, defaults to np.random.

    Returns:
        points_coord (np.ndarray): (L, 3). Sampled 3D points.
        barycentric_coord (np.ndarray): (L, 3). Barycentric coords.
        indices (np.ndarray): (L,). Face index for each point.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Triangle vertices (M,3,3)
    tri_vertices = vertices[faces]

    # Triangle areas (M,)
    v0 = tri_vertices[:, 1] - tri_vertices[:, 0]
    v1 = tri_vertices[:, 2] - tri_vertices[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(v0, v1), axis=1)

    # Number of samples per face
    n_samples_per_face = np.ceil(areas * points_per_area).astype(int)
    total_samples = n_samples_per_face.sum()
    if total_samples == 0:
        return (
            np.zeros((0, 3)),
            np.zeros((0, 3)),
            np.zeros((0,), dtype=int),
        )

    # Assign each sample a face id (L,)
    indices = np.repeat(np.arange(len(faces)), n_samples_per_face)

    # Random barycentric (L,2) -> convert to (L,3)
    u = rng.random(total_samples)
    v = rng.random(total_samples)
    mask = u + v > 1
    u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
    w = 1 - (u + v)
    barycentric_coord = np.stack([w, u, v], axis=1)

    # Gather triangle vertices for each sample (L,3,3)
    sampled_tris = tri_vertices[indices]

    # Weighted sum -> points (L,3)
    points_coord = np.einsum("lj,ljk->lk", barycentric_coord, sampled_tris)

    return points_coord, barycentric_coord, indices


def map_to_uv(
    uvs: np.ndarray,
    faces_unwrapped: np.ndarray,
    barycentric_coord: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Vectorized barycentric interpolation in UV space.

    Args:
        uvs (np.ndarray): (V, 2) UV coordinates.
        faces_unwrapped (np.ndarray): (M, 3) indices into uvs.
        barycentric_coord (np.ndarray): (L, 3) barycentric weights.
        indices (np.ndarray): (L,) face index ids.

    Returns:
        sample_uvs (np.ndarray): (L, 2).
    """
    # Triangle uv coords (M,3,2)
    tri_uvs = uvs[faces_unwrapped]

    # Gather triangles for samples (L,3,2)
    sampled_tris = tri_uvs[indices]

    # Weighted sum -> (L,2)
    sample_uvs = np.einsum("lj,ljk->lk", barycentric_coord, sampled_tris)
    return sample_uvs


def query_scalar_field(points_coord: np.ndarray, data_points: np.ndarray):
    """
    Query scalar field f(x,y,z) in vectorized form.

    Args:
        points_coord (np.ndarray): (L, 3).
        data_points (np.ndarray): (data_len, 4) xyz,value

    Returns:
        values (np.ndarray): (L,).
    """
    kdtree = KDTree(data_points[:, :3])
    _, nearest_indices = kdtree.query(
        points_coord, k=1
    )  # TODO: check if one should consider the pruned zero-value points. For example, if the nearest point is zero-value (and pruned), this func may look for the next non-zero nearest value, which is not intended.
    interpolated_values = data_points[nearest_indices, 3]
    return interpolated_values


def bake_to_texture(
    sample_uvs: np.ndarray,
    values: np.ndarray,
    H: int,
    W: int,
):
    """
    Bake scalar values into a texture map using scatter-add.

    Args:
        sample_uvs (np.ndarray): (L, 2) in [0,1].
        values (np.ndarray): (L,).
        H (int): texture height.
        W (int): texture width.

    Returns:
        texture (np.ndarray): (H, W).
    """
    tex_sum = np.zeros((H, W), dtype=float)
    tex_count = np.zeros((H, W), dtype=int)

    # UV -> pixel (L,)
    us = np.clip((sample_uvs[:, 0] * (W - 1)).astype(int), 0, W - 1)
    vs = np.clip((sample_uvs[:, 1] * (H - 1)).astype(int), 0, H - 1)

    # Scatter-add
    np.add.at(tex_sum, (vs, us), values)
    np.add.at(tex_count, (vs, us), 1)

    # Normalize
    mask = tex_count > 0
    tex_sum[mask] /= tex_count[mask]

    return tex_sum


def nearest_fill(texture):
    # mask: 1 for missing (0), 0 for valid
    mask = texture == 0

    dist, indices = distance_transform_edt(mask, return_indices=True)

    # Fill with nearest value
    filled = texture[tuple(indices)]
    return filled


def linear_fill(texture):
    h, w = texture.shape
    y, x = np.mgrid[0:h, 0:w]

    known = texture > 0
    coords = np.column_stack((x[known], y[known]))
    values = texture[known]

    filled = griddata(coords, values, (x, y), method="linear", fill_value=0)
    return filled


def denoise_texture(texture, method="linear"):
    """
    Fill missing (zero) values in a sparse 2D texture map using interpolation.

    Args:
        texture (numpy.ndarray):
            A 2D NumPy array representing the texture map.
            Zero entries are treated as missing data to be filled.
        method (str, optional):
            Interpolation method to use. Options are:
            - "linear": Fill using linear interpolation over known non-zero points.
            - "nearest": Fill using nearest-neighbor interpolation.
            Defaults to "linear".

    Returns:
        numpy.ndarray:
            A 2D NumPy array of the same shape as `texture`,
            with missing (zero) values replaced by interpolated values.

    Raises:
        ValueError: If `method` is not one of {"linear", "nearest"}.
    """
    if method == "linear":
        return linear_fill(texture)
    elif method == "nearest":
        return nearest_fill(texture)
    else:
        raise ValueError(f"Invalid method: {method}. Must be one of 'linear' or 'nearest'.")


def generate_slice_mesh(center: np.ndarray, h: int, w: int, rh: int, rw: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a slice plane mesh in the XY-plane, centered on `center`.

    Args:
        center (np.ndarray): 3D coordinate where the center of the mesh will be placed (shape (3,))
        h (int): physical height of the plane
        w (int): physical width of the plane
        rh (int): resolution along height (number of vertices)
        rw (int): resolution along width (number of vertices)

    Returns:
        tuple[np.ndarray, np.ndarray]:
            vertices: (N, 3) array of 3D coordinates
            faces: (M, 3) array of integer indices into vertices
    """
    # Generate grid in local XY-plane
    y = np.linspace(-h / 2, h / 2, rh)
    x = np.linspace(-w / 2, w / 2, rw)
    xx, yy = np.meshgrid(x, y)

    vertices = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1) + center

    # Vectorized face construction
    # indices grid
    idx = np.arange(rh * rw).reshape(rh, rw)

    # Lower-left corners of each quad
    v0 = idx[:-1, :-1].ravel()
    v1 = idx[:-1, 1:].ravel()
    v2 = idx[1:, :-1].ravel()
    v3 = idx[1:, 1:].ravel()

    # 2 triangles per quad
    faces = np.stack([np.column_stack([v0, v1, v2]), np.column_stack([v1, v3, v2])], axis=1).reshape(-1, 3)

    return vertices, faces
