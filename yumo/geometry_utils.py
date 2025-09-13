import hashlib
import logging

import cv2
import numpy as np
import xatlas
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from yumo.utils import profiler

logger = logging.getLogger(__name__)
_tree_cache = {}


def unwrap_uv(
    vertices: np.ndarray, faces: np.ndarray, padding: int = 16, brute_force: bool = False
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs UV unwrapping for a given 3D mesh using the xatlas library with
    padding settings to reduce UV bleeding.

    Args:
        vertices (np.ndarray): (N, 3) float array of mesh vertex positions.
        faces (np.ndarray): (M, 3) int array of triangular face indices.
        padding (int): Padding in pixels between UV islands (default=16px).
        brute_force (bool): Slower, but gives the best result. If false, use random chart placement.

    Returns:
        Tuple containing:
        - param_corner (np.ndarray): (M*3, 2) UV coords per face corner.
        - texture_height (int): Atlas height in pixels.
        - texture_width (int): Atlas width in pixels.
        - vmapping (np.ndarray): (V,) mapping of unwrapped vertex → original vertex.
        - faces_unwrapped (np.ndarray): (M, 3) face indices into unwrapped verts.
        - uvs (np.ndarray): (V, 2) UV coords ∈ [0, 1].
        - vertices_unwrapped (np.ndarray): (V, 3) unwrapped vertex positions.
    """
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)

    # Chart options (optional - default usually works fine)
    chart_options = xatlas.ChartOptions()

    # Pack with resolution & padding to avoid bleeding
    pack_options = xatlas.PackOptions()
    pack_options.padding = padding
    pack_options.bilinear = True
    pack_options.rotate_charts = True
    pack_options.bruteForce = brute_force
    atlas.generate(chart_options=chart_options, pack_options=pack_options)

    # Get unwrapped data
    vmapping, faces_unwrapped, uvs = atlas[0]  # [N], [M, 3], [N, 2]
    vertices_unwrapped = vertices[vmapping]

    # Flatten per-corner UVs for APIs like OpenGL
    param_corner = uvs[faces_unwrapped].reshape(-1, 2)

    texture_height, texture_width = atlas.height, atlas.width

    return (
        param_corner,
        texture_height,
        texture_width,
        vmapping,
        faces_unwrapped,
        uvs,
        vertices_unwrapped,
    )


def uv_mask(
    uvs: np.ndarray,
    faces_unwrapped: np.ndarray,
    texture_width: int,
    texture_height: int,
    dilation: int = 2,
    supersample: int = 4,
) -> np.ndarray:
    """
    Creates a binary (or soft) mask indicating which parts of the texture atlas
    are occupied by the mesh (True) and which are empty (False).

    Args:
        uvs: (N,2) UV coordinates in [0,1].
        faces_unwrapped: (F,3) triangle indices into uvs.
        texture_width: target texture width in pixels.
        texture_height: target texture height in pixels.
        dilation: number of pixels to expand the mask after rasterization.
        supersample: supersampling factor for more accurate rasterization.
    """
    # --- supersample resolution for smoother edges ---
    hi_w = texture_width * supersample
    hi_h = texture_height * supersample

    # Convert UVs to high-res pixel coords
    uv_pixels = np.zeros_like(uvs, dtype=np.float64)
    uv_pixels[:, 0] = uvs[:, 0] * (hi_w - 1)
    uv_pixels[:, 1] = (1.0 - uvs[:, 1]) * (hi_h - 1)  # flip Y

    # Initialize high-res mask
    hi_mask = np.zeros((hi_h, hi_w), dtype=np.uint8)

    # Rasterize triangles
    for face in faces_unwrapped:
        pts = uv_pixels[face].astype(np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(hi_mask, pts, 255)  # type: ignore[call-overload]

    # Downsample back to target resolution with area interpolation
    mask = cv2.resize(hi_mask, (texture_width, texture_height), interpolation=cv2.INTER_AREA)

    # Normalize to [0,1] float mask (soft edges preserved)
    mask = mask.astype(np.float64) / 255.0

    # Optional dilation to pad seams (applied on final resolution)
    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        hard_mask = (mask > 0).astype(np.uint8) * 255
        dilated = cv2.dilate(hard_mask, kernel, iterations=1)
        mask = np.maximum(mask, dilated.astype(np.float64) / 255.0)

    return mask


def triangle_areas(tri_vertices: np.ndarray) -> np.ndarray:
    # Triangle areas (M,)
    v0 = tri_vertices[:, 1] - tri_vertices[:, 0]
    v1 = tri_vertices[:, 2] - tri_vertices[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(v0, v1), axis=1)
    return areas  # type: ignore[no-any-return]


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

    areas = triangle_areas(tri_vertices)

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
    return sample_uvs  # type: ignore[no-any-return]


def get_tree(data_points: np.ndarray):
    key = hashlib.sha256(data_points.view(np.uint8)).hexdigest()
    if key not in _tree_cache:
        _tree_cache[key] = cKDTree(data_points[:, :3])
    return _tree_cache[key]


def clear_tree_cache():
    _tree_cache.clear()


@profiler(profiler_logger=logger)
def query_scalar_field(points_coord: np.ndarray, data_points: np.ndarray, cache: bool = True) -> np.ndarray:
    """
    Query scalar field f(x,y,z) in vectorized form.

    Args:
        points_coord (np.ndarray): (L, 3).
        data_points (np.ndarray): (data_len, 4) xyz,value
        cache (bool): Enable cache for cKDTree construction.

    Returns:
        values (np.ndarray): (L,).
    """
    kdtree = get_tree(data_points) if cache else cKDTree(data_points[:, :3])

    _, nearest_indices = kdtree.query(points_coord, k=1)

    interpolated_values = data_points[nearest_indices, 3]
    return interpolated_values  # type: ignore[no-any-return]


@profiler(profiler_logger=logger)
def bake_to_texture(
    sample_uvs: np.ndarray,
    values: np.ndarray,
    H: int,
    W: int,
):
    """
    Bake scalar values into a texture map using scatter-add.

    UV space (0,0 bottom-left) is converted to image space (0,0 top-left).
    """
    tex_sum = np.zeros((H, W), dtype=float)
    tex_count = np.zeros((H, W), dtype=int)

    # UV -> pixel coords (with vertical flip to align with image convention)
    us = np.clip((sample_uvs[:, 0] * (W - 1)).astype(int), 0, W - 1)
    vs = np.clip(((1.0 - sample_uvs[:, 1]) * (H - 1)).astype(int), 0, H - 1)

    # Scatter values
    np.add.at(tex_sum, (vs, us), values)
    np.add.at(tex_count, (vs, us), 1)

    # Normalize
    mask = tex_count > 0
    tex_sum[mask] /= tex_count[mask]

    return tex_sum


def nearest_fill(
    texture,
    max_dist=16,
    mask: np.ndarray = None,  # type: ignore[assignment]
    **kwargs,
):  # kwargs for compatibility
    """

    Args:
        texture:
        max_dist:
        mask: (0) Padding (1) Islands.
        **kwargs:

    Returns:

    """
    # mask: 1 for missing (0), 0 for valid
    zeros_mask = texture == 0
    if mask is not None:
        zeros_mask = zeros_mask & mask.astype(bool)  # only fill islands

    dist, indices = distance_transform_edt(zeros_mask, return_indices=True)

    # Fill with nearest value
    filled = texture[tuple(indices)]
    filled[dist > max_dist] = 0  # drop far-away fills

    return filled


def nearest_and_blur(
    texture,
    blur_sigma=1.0,
    max_dist=16,
    mask: np.ndarray = None,  # type: ignore[assignment]
    **kwargs,
) -> np.ndarray:  # max dist should be smaller than the padding in unwrap_uv
    """

    Args:
        texture:
        blur_sigma:
        max_dist:
        mask: (0) Padding (1) Islands.
        **kwargs:

    Returns:

    """
    nn_result = nearest_fill(texture, max_dist=max_dist, mask=mask, **kwargs)
    kwargs.pop("mask", None)
    kwargs.pop("sigma", None)
    return blur(nn_result, sigma=blur_sigma, mask=mask, **kwargs)


def blur(
    texture: np.ndarray,
    sigma: float = 1.0,
    mask: np.ndarray = None,  # type: ignore[assignment]
    **kwargs,
) -> np.ndarray:
    """Apply Gaussian blur to a texture while ignoring zero-padding.

    This function performs a mask-aware Gaussian filter: both the texture
    and a binary mask are smoothed, and then the results are normalized
    to prevent padded zeros from biasing the blur.

    Args:
        texture (np.ndarray): Input texture array. Padding regions should be zeros.
            Can be 2D (H, W) or 3D (H, W, C).
        sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.0.
        mask (np.ndarray): (0) Padding (1) Islands.
        **kwargs: Useless. For compatibility only.

    Returns:
        np.ndarray: Blurred texture with padding ignored. Same shape as input.
    """
    # Build binary mask
    zeros_mask = texture != 0  # we don't & mask here, as holes inside islands must be normalized too
    zeros_mask = zeros_mask.astype(np.float64)

    # Apply Gaussian filter
    blurred_tex = gaussian_filter(texture.astype(np.float64), sigma=sigma)
    blurred_mask = gaussian_filter(zeros_mask.astype(np.float64), sigma=sigma)

    # Normalize
    eps = 1e-8
    result = blurred_tex / (blurred_mask + eps)

    # Set padding back to zero
    result[mask == 0] = 0

    return result  # type: ignore[no-any-return]


@profiler(profiler_logger=logger)
def denoise_texture(texture, method="nearest_and_gaussian", **kwargs):
    """
    Fill missing (zero) values in a sparse 2D texture map using interpolation.

    Args:
        texture (numpy.ndarray):
            A 2D NumPy array representing the texture map.
            Zero entries are treated as missing data to be filled.
        method (str, optional):
            Interpolation method to use. Options are:
            - "gaussian": Simple gaussian filter.
            - "nearest_and_gaussian": Fill using nearest-neighbour then gaussian blur.
            - "nearest": Fill using nearest-neighbor interpolation.
            Defaults to "linear".

    Returns:
        numpy.ndarray:
            A 2D NumPy array of the same shape as `texture`,
            with missing (zero) values replaced by interpolated values.

    Raises:
        ValueError: If `method` is not one of {"linear", "nearest"}.
    """
    if method == "gaussian":
        return blur(texture, **kwargs)
    elif method == "nearest_and_gaussian":
        return nearest_and_blur(texture, **kwargs)
    elif method == "nearest":
        return nearest_fill(texture, **kwargs)
    else:
        raise ValueError(f"Invalid method: {method}. Must be one of 'linear' or 'nearest'.")


def generate_slice_mesh(center: np.ndarray, h: float, w: float, rh: int, rw: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a slice plane mesh in the XY-plane, centered on `center`.

    Args:
        center (np.ndarray): 3D coordinate where the center of the mesh will be placed (shape (3,))
        h (float): physical height of the plane
        w (float): physical width of the plane
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
