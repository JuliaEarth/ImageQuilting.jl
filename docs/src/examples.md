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

problem = SimulationProblem(CartesianGrid(200,200), :facies => Int, 3)

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
problem = SimulationProblem(CartesianGrid(200,200), :Z => Int, 3)

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

It is possible to incorporate auxiliary variables to
guide the selection of patterns from the training image.

```@example basics
using ImageFiltering

# image assumed as ground truth (unknown)
truthimg = geostatsimage("WalkerLakeTruth")

# training image with similar patterns
trainimg = geostatsimage("WalkerLake")

plot(plot(trainimg), plot(truthimg))
```

```@example basics
# forward model (blur filter)
function forward(data)
    img = asarray(data, :Z)
    krn = KernelFactors.IIRGaussian([10,10])
    fwd = imfilter(img, krn)
    georef((fwd=fwd,), domain(data))
end

# apply forward model to both images
data   = forward(truthimg)
dataTI = forward(trainimg)

plot(plot(dataTI), plot(data))
```

```@example basics
# simulate patterns over the domain of interest
problem = SimulationProblem(domain(truthimg), :Z => Float64, 3)

solver = IQ(
    :Z => (
        trainimg = trainimg,
        tilesize = (27,27),
        soft = (data,dataTI)
    )
)

solution = solve(problem, solver)

plot(solution)
```