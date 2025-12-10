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
using CairoMakie

TI₁ = geostatsimage("Strebelle")
TI₂ = geostatsimage("StoneWall")

timg₁ = reshape(TI₁.facies, size(domain(TI₁)))
timg₂ = reshape(TI₂.Z, size(domain(TI₂)))

voxelreuseplot(timg₁)
voxelreuseplot!(timg₂)
```
![Voxel reuse plot](images/voxelreuse.png)
