!!! note

    Consider installing the [GeoStatsImages.jl](https://github.com/juliohm/GeoStatsImages.jl) package.

## Unconditional

An example of unconditional simulation (i.e. simulation without data).
This is the original Efros-Freeman algorithm for texture synthesis with a
few additional options.

```julia
using ImageQuilting
using GeoStatsImages

TI = training_image("Strebelle")[:,:,1]
reals = iqsim(TI, (62, 62), nreal=3)

TI = training_image("StoneWall")[:,:,1]
reals, cuts, voxs = iqsim(TI, (13, 13), nreal=3, debug=true)
```
![Unconditional simulation](images/unconditional.png)

## Hard data

Hard data (i.e. point data) can be honored during simulation. This can be useful
when images represent a spatial property of a physical domain. For example, the
subsurface of the Earth is only known with certainty at a few well locations.

```julia
using ImageQuilting
using GeoStatsImages

TI = training_image("Strebelle")[:,:,1]

data = Dict(
  CartesianIndex(50,50)   => 1,
  CartesianIndex(190,50)  => 0,
  CartesianIndex(150,170) => 1,
  CartesianIndex(150,190) => 1
)

reals, cuts, voxs = iqsim(TI, (30, 30), hard=data, debug=true)
```
![Hard data conditioning](images/hard.gif)

![Hard data conditioning](images/hard.png)

## Soft data

Sometimes it is also useful to incorporate auxiliary variables defined in the
domain, which can guide the selection of patterns in the training image. This
example shows how to achieve this texture transfer efficiently.

```julia
using ImageQuilting
using GeoStatsImages
using ImageFiltering

TI    = training_image("WalkerLake")[:,:,1]
truth = training_image("WalkerLakeTruth")[:,:,1]

G(m) = imfilter(m, KernelFactors.IIRGaussian([10,10]))

data   = G(truth)
dataTI = G(TI)

reals = iqsim(TI, (27, 27), size(truth), soft=[(data,dataTI)], nreal=3)
```
![Soft data conditioning](images/soft.png)

## Masked grids

Voxels marked with the special symbol `NaN` are treated as inactive. The algorithm
will skip tiles that only contain inactive voxels to save computation and will
generate realizations that are consistent with the mask. This is particularly
useful with complex 3D models that have large inactive portions.

```julia
using ImageQuilting
using GeoStatsImages

TI = training_image("Strebelle")[:,:,1]

# skip circle at the center
nx, ny = size(TI)
r = 100; circle = []
for i=1:nx, j=1:ny
    if (i-nx÷2)^2 + (j-ny÷2)^2 < radius^2
        push!(circle, CartesianIndex(i,j)=>NaN)
    end
end

reals = iqsim(TI, (62, 62), hard=Dict(circle), nreal=3)
```
![Masked grids](images/masked.png)
