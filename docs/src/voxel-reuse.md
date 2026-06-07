## Helper function

A helper function is provided for the fast approximation of the *mean voxel reuse*:

```@docs
voxelreuse
```

## Plot recipe

A plot recipe is provided for tile design in image quilting:

```@docs
voxelreuseplot
```

In order to plot the voxel reuse of a training image, install any of the
[Makie.jl](https://docs.makie.org) backends.

```julia
] add CairoMakie
```

The example below uses training images from the
[GeoStatsImages.jl](https://github.com/JuliaEarth/GeoStatsImages.jl) package:

```julia
using ImageQuilting
using GeoStatsImages
import GLMakie as Mke

img1 = geostatsimage("Strebelle")
img2 = geostatsimage("StoneWall")

trainimg1 = reshape(img1.facies, (size(domain(img1))..., 1))
trainimg2 = reshape(img2.Z, (size(domain(img2))..., 1))

voxelreuseplot(trainimg1)
voxelreuseplot!(trainimg2, color=:salmon)
```
![Voxel reuse plot](images/voxelreuse.png)
