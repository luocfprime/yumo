# Usage Guide

!!! abstract

    This tool consists of two parts:

    1. `yumo prune`: Removes data points inside the mesh model.
    2. `yumo viz`: Interactive visualization.

## Internal Point Pruning

!!! warning "Note"

    The provided mesh must be watertight, otherwise it's impossible to calculate which data points fall inside the mesh.

Use `yumo prune -h` to view detailed help.

## Visualization

The visualization contains three main components and several smaller widgets.

1. 3D data point cloud visualization: Visualizes Tecplot `.plt` file point clouds.
2. Spatial slice visualization: Views the distribution of 3D scalar fields on slices.
3. Mesh surface field strength visualization: Visualizes the distribution of 3D scalar fields on the mesh surface.

Use `yumo viz -h` to view command line parameters.

E.g.

```bash
yumo viz --data xxxx.plt --mesh yyyy.STL
```

### Input Data Transformation

If you want to visualize data with a log scale, you can use the input data transformation feature by specifying the transformation method with `--prep`.

Example: Input data log scale transformation, where zeros in the original data will be replaced with the minimum value after transformation.

```bash
yumo viz --data xxxx.plt --mesh yyyy.STL --prep log_10
```

### Point Cloud

!!! info "Data Rendering Process"

    The original data is directly colored according to the colormap. Based on the maximum and minimum values of the colormap, parts exceeding the threshold will be rendered according to the maximum value color.

![point cloud gui]([[url.prefix]]/media/points.jpg)

1. Whether to display (displaying too many point clouds will cause lag, you can turn off point cloud display or adjust the display quantity through 4).
2. Point diameter.
3. Rendering method: sphere has better effect but higher performance cost, quad is rougher but has less performance cost.
4. Display threshold: Only points with values greater than this threshold will be displayed. You can double-click to input or drag to adjust.

### Slices

!!! info "Data Rendering Process"

    1. Calculate slice sampling points according to the set width, height, and resolution. Uniform grid sampling.
    2. Based on the slice sampling points, **find the nearest data point** in space using nearest neighbor search.
    3. **Directly assign** the data point value to the sampling point on the slice.
    4. Render the sampling points as grid vertices of the slice. Polyscope will automatically interpolate between grid vertices.

![slice gui]([[url.prefix]]/media/slice.jpg)

1. Click to create a slice.
2. Check to display the slice.
3. Live mode: Every time you drag the position/angle of the slice, the slice rendering result updates in real-time. Checking this may cause lag.
4. Gizmo: A small tool used to adjust the position/angle of the slice.
5. Destroy: Click to delete the slice.
6. Compute: Calculate the slice result. (When Live mode is False, you need to click to get the slice result)
7. Transparency adjustment.
8. Slice height.
9. Slice width.
10. Slice height resolution.
11. Slice width resolution.

### Mesh

!!! info "Data Rendering Process"

    1. Sample points with random equal surface density based on the set average number of sampling points per unit area (sampling points are mesh surface sample points, which are 3D coordinates). Unit area refers to the unit area of the mesh surface.
    2. Based on the sampling points, **find the nearest data point** in space using nearest neighbor search.
    3. **Directly assign** the data point value to the sampling point.
    4. Write the sampling points back to the texture according to UV mapping (if the same texture pixel is recorded multiple times, the average will be taken). **When writing back, the UV coordinates are directly rounded for scattering, so there will be small discrete errors.**
    5. (Optional) The current texture is the original texture data, which may be very sparse. You can choose to denoise for appropriate smoothing.

    Denoising methods:

    - nearest_and_gaussian: First perform nearest neighbor pixel search with a maximum of Max Dist for pixels on the texture (Max Dist avoids UV bleeding), then apply Gaussian blur with Sigma.
    - nearest: Only perform nearest neighbor pixel search with a maximum of Max Dist for pixels on the texture.
    - gaussian: Apply Gaussian blur with a maximum of Max Dist to pixels on the texture.

![mesh gui]([[url.prefix]]/media/mesh.jpg)

1. Check to display the mesh.
2. Sampling rate.
3. Bake: Execute the process of sampling and assigning values to the texture.
4. Material: Built-in blended texture of "flat" and "clay". The larger the flat coefficient, the lighting effect will be weaker, and the color of the mesh will be more close to the colormap (with less shadow etc.).
5. Check to enable texture denoising.
6. Denoising method.
7. (Optional) Maximum distance of the nearest neighbor pixel. (Appropriately reduce to avoid UV bleeding)
8. (Optional) Variance of Gaussian smoothing.

!!! warning "Note"

    When dragging the sampling rate, the current sampling points on the surface will be visualized. ==**At this time, the displayed sampling points are not real data, but the maximum value of the color map.**==
    Only after clicking the Bake button will the process of sampling and assigning values to the texture be actually executed.

    <video width=100% autoplay muted loop>
      <source src="[[url.videos]]/media/movies/surface_sampling.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>

### Coord Picker

This feature allows you to click on any position on the screen to get the coordinate point corresponding to the current screen coordinate and query the field strength value corresponding to the coordinate point.
You can collapse and expand the menu to turn off and on this feature. It is off by default.

![coord picker]([[url.prefix]]/media/coord_picker.jpg)

When Query Field is checked, clicking on any non-empty position on the screen will use **nearest neighbor search** to get the point cloud data point corresponding to the clicked coordinates.

When clicking on the mesh surface, Coord Picker will additionally query the surface field strength value recorded in the texture data through UV coordinates.
When Query Field is also enabled, it will calculate the relative error between the queried texture record field strength value and the point cloud queried field strength value: e = |f_texture - f_pointcloud| / |f_pointcloud|.

### Color Bar

![color control]([[url.prefix]]/media/color_control.jpg)

1. Colormap.
2. Visualization minimum value. (Colors below the minimum value will be uniformly colored as the minimum value color)
3. Visualization maximum value. (Colors above the maximum value will be uniformly colored as the maximum value color)
4. Reset range.

!!! warning "Note"

    If the color bar is not displaying properly:

    1. Check the Polyscope's left-side Structures menu to see if it's not collapsed.
    2. Check Structures/Floating Quantities to see if there is a colorbar.
    3. If there is a colorbar, click on it and check if enabled is checked.
    4. Adjust the screen size to see if the colorbar is a floating small window (it may be collapsed). If so, you can drag a corner of the window to adjust its size.

### Floating Window

![floating quantities]([[url.prefix]]/media/floating.jpg)

1. Color bar.
2. Original texture: The texture after sampling + UV mapping.
3. Denoised texture: The texture obtained after denoising 2.
