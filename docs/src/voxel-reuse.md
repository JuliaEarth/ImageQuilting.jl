## Helper function

A helper function is provided for the fast approximation of the *mean voxel reuse*:

```julia
mean, dev = voxelreuse(training_image::AbstractArray,
                       tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;
                       overlapx=1/6, overlapy=1/6, overlapz=1/6,
                       cut=:boykov, simplex=false, nreal=10,
                       threads=CPU_PHYSICAL_CORES, gpu=false)
```

with `mean` in the interval ``[0,1]`` and `dev` the standard deviation. The approximation
gets better as `nreal` is made larger.

## Plot recipe

A plot recipe is provided for template design in image quilting. In order to plot the voxel
reuse of a training image, install [Plots.jl](https://github.com/JuliaPlots/Plots.jl) and
any of its supported backends (e.g. [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl)):

```julia
Pkg.add("Plots")
Pkg.add("PyPlot")
```

The example below uses training images from the
[GeoStatsImages.jl](https://github.com/juliohm/GeoStatsImages.jl) package:

```julia
using ImageQuilting
using GeoStatsImages
using Plots

TI₁ = training_image("Strebelle")
TI₂ = training_image("StoneWall")

plot(VoxelReuse(TI₁), label="Strebelle")
plot!(VoxelReuse(TI₂), label="StoneWall")
```
![Voxel reuse plot](images/voxelreuse.png)
