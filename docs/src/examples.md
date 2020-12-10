## Basic

An example of unconditional simulation (i.e. simulation without data).
This is the original Efros-Freeman algorithm for texture synthesis with a
few additional options.

```@example basics
using GeoStats
using ImageQuilting
using GeoStatsImages
using Plots
gr(size=(850,300)) # hide

problem = SimulationProblem(RegularGrid(200,200), :facies => Int, 3)

solver = IQ(
    :facies => (
        trainimg = geostatsimage("Strebelle"),
        tilesize = (62,62)
    )
)

solution = solve(problem, solver)

plot(solution)
```

```@example basics
problem = SimulationProblem(RegularGrid(200,200), :Z => Int, 3)

solver = IQ(
    :Z => (
        trainimg = geostatsimage("StoneWall"),
        tilesize = (13,13)
    )
)

solution = solve(problem, solver)

plot(solution)
```

## Hard data

Hard data (i.e. point data) can be honored during simulation.

```@example basics
trainimg = geostatsimage("Strebelle")
observed = sample(trainimg, 20, replace=false)

problem = SimulationProblem(observed, domain(trainimg), :facies, 3)

solver = IQ(
    :facies => (
        trainimg = trainimg,
        tilesize = (30,30)
    )
)

solution = solve(problem, solver)

plot(solution)
```

## Masked grids

Voxels marked with the special symbol `NaN` are treated as inactive. The algorithm
will skip tiles that only contain inactive voxels to save computation and will
generate realizations that are consistent with the mask. This is particularly
useful with complex 3D models that have large inactive portions.

```@example basics
trainimg = geostatsimage("Strebelle")

# skip circle at the center
nx, ny = size(domain(trainimg))
r = 100; circle = []
for i=1:nx, j=1:ny
    if (i-nx÷2)^2 + (j-ny÷2)^2 < r^2
        push!(circle, CartesianIndex(i,j))
    end
end

problem = SimulationProblem(domain(trainimg), :facies => Float64, 3)

solver = IQ(
    :facies => (
        trainimg = trainimg,
        tilesize = (62,62),
        inactive = circle
    )
)

solution = solve(problem, solver)

plot(solution)
```

## Soft data

Sometimes it is also useful to incorporate auxiliary variables to
guide the selection of patterns in the training image.

```julia
using ImageQuilting
using GeoStatsImages
using ImageFiltering

TI    = geostatsimage("WalkerLake")
truth = geostatsimage("WalkerLakeTruth")

G(m) = imfilter(m, KernelFactors.IIRGaussian([10,10]))

data   = G(truth)
dataTI = G(TI)

reals = iqsim(TI, (27, 27), size(truth), soft=[(data,dataTI)], nreal=3)
```
![Soft data conditioning](images/soft.png)
