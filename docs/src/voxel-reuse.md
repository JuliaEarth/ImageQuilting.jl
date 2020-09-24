## Helper function

A helper function is provided for the fast approximation of the *mean voxel reuse*:

```@docs
voxelreuse
```

## Plot recipe

A plot recipe is provided for tile design in image quilting. In order to plot the voxel
reuse of a training image, install [Plots.jl](https://github.com/JuliaPlots/Plots.jl) and
any of its supported backends (e.g. [GR.jl](https://github.com/jheinen/GR.jl)):

```julia
] add Plots
```

The example below uses training images from the
[GeoStatsImages.jl](https://github.com/JuliaEarth/GeoStatsImages.jl) package:

```julia
using ImageQuilting
using GeoStatsImages
using Plots

TI₁ = geostatsimage("Strebelle")
TI₂ = geostatsimage("StoneWall")

voxelreuseplot(TI₁, label="Strebelle")
voxelreuseplot!(TI₂, label="StoneWall")
```
![Voxel reuse plot](images/voxelreuse.png)
