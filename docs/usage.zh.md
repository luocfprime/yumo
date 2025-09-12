# 使用说明

!!! abstract

    本工具包含两部分：

    1. `yumo prune`：剪掉数据点云中在mesh模型内部的点。
    2. `yumo vis`：交互式可视化。

## 内部点裁剪

!!! warning "注意"

    提供的mesh必须是watertight的，否则无法计算哪些数据点落在了mesh内部。

使用`yumo prune -h`查看详细帮助。

## 可视化

可视化包含三个主要部件以及若干小部件。

1. 3D数据点云可视化：可视化Tecplot `.plt`文件点云。
2. 空间切面可视化：查看3D标量场在切面上的分布。
3. mesh表面场强分布可视化：可视化3D标量场在mesh的表面分布。

使用`yumo vis -h`查看命令行参数。

E.g.

```bash
yumo viz --data xxxx.plt --mesh yyyy.STL
```

### 点云

!!! info "数据渲染过程"

    直接对原始数据按照colormap进行着色。根据colormap的最大最小值，超出阈值部分会按照最大值颜色进行渲染。

![point cloud gui]([[url.prefix]]/media/points.jpg)

1. 是否显示（显示点云数量过多会卡顿，可以关闭点云显示或通过4调整显示数量）。
2. 点直径。
3. 渲染方式：sphere效果好性能开销大，quad粗糙性能开销少。
4. 显示阈值：数值大于该阈值才显示该点。可以双击输入或拖动调整。

### 切面

!!! info "数据渲染过程"

    1. 根据设定宽高以及分辨率计算切面采样点。均匀网格采样。
    2. 根据切面采样点，**最近邻查找**空间中最近数据点。
    3. 将数据点**直接赋值**给切面上的采样点。
    4. 将采样点视为切面的网格顶点进行渲染。Polyscope会自动进行网格顶点之间的插值。

![slice gui]([[url.prefix]]/media/slice.jpg)

1. 点击创建切面。
2. 勾选显示切面。
3. Live模式：每拖动一下切面的位置/角度，切面渲染结果实时更新。勾选后可能造成卡顿。
4. Gizmo：小工具，用于调整切面位置/角度。
5. Destroy：点击后删除切面。
6. Compute：计算切面结果。（Live模式为False时，需要点击才能得到切面结果）
7. 透明度调整。
8. 切面高度。
9. 切面宽度。
10. 切面高度分辨率。
11. 切面宽度分辨率。

### mesh

!!! info "数据渲染过程"

    1. 根据设定平均单位面积采样点个数，进行随机等表面密度采样（采样点为mesh表面样本点，是3D坐标）。单位面积指mesh表面积的单位面积。
    2. 根据采样点，**最近邻查找**空间中最近数据点。
    3. 将数据点**直接赋值**给采样点。
    4. 将采样点根据UV映射，写回到texture中（同一个纹理像素如果多次记录，会最终取平均）。**写回的时候直接对UV坐标进行取整scatter，因此会有微小的离散误差。**
    5. （可选）当前纹理是原始纹理数据，可能会非常稀疏。可以选择去噪进行适当平滑。

    去噪方式：

    - nearest_and_gaussian: 先对texture上的像素进行最大为Max Dist的最近邻像素查找（Max Dist避免UV bleeding），然后施加Sigma的高斯模糊。
    - nearest：只对texture上的像素进行最大为Max Dist的最近邻像素查找。
    - gaussian：对texture上的像素进行最大为Max Dist的高斯模糊。

![mesh gui]([[url.prefix]]/media/mesh.jpg)

1. 勾选显示mesh。
2. 采样率。
3. Bake：执行采样、赋值给texture的过程。
4. 勾选启用材质去噪。
5. 去噪方法。
6. （可选）最近邻的像素的最大距离。（适当调小避免UV bleeding）
7. （可选）高斯平滑的方差。

!!! warning "注意"

    当拖动采样率时，会可视化当前表面上的采样点。==**此时现实的采样点不是真实数据，而是color map的最大值。**==
    只有点击Bake按钮后，才会真正执行采样、赋值给texture的过程。

    <video width=100% autoplay muted loop>
      <source src="[[url.prefix]]/media/movies/surface_sampling.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>

### 颜色条

![color control]([[url.prefix]]/media/color_control.jpg)

1. Colormap。
2. 可视化最小值。（低于最小值的颜色统一为最小值的颜色）
3. 可视化最大值。（高于最大值的颜色统一为最大值的颜色）
4. 重制范围。

!!! warning "注意"

    如果颜色条没有正常显示：

    1. 查看Polyscope的左侧Structures菜单，看看是否没有被折叠。
    2. 查看Structures/Floating Quantities，查看是否有colorbar。
    3. 如果有colorbar，点开，查看enabled是否勾选。
    4. 调整屏幕尺寸，看看colorbar是不是一个悬浮的小窗口（可能被折叠）。如果是，可以拖动窗口一角调节大小。

### 悬浮窗口

![floating quantities]([[url.prefix]]/media/floating.jpg)

1. 颜色条。
2. 原始材质：经过采样+UV映射后的材质。
3. 去噪后的材质：2经过去噪得到的材质。
