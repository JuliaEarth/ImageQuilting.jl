Consider installing the [GeoStatsImages.jl](https://github.com/juliohm/GeoStatsImages.jl) package.

# Unconditional

```julia
using ImageQuilting
using GeoStatsImages

TI = training_image("Strebelle")
reals = iqsim(TI, 62, 62, 1, size(TI)..., nreal=3)

TI = training_image("StoneWall")
reals, cuts, voxs = iqsim(TI, 13, 13, 1, size(TI)..., nreal=3, debug=true)
```

# Hard data

TODO

# Soft data

```julia
using ImageQuilting
using GeoStatsImages
using Images: imfilter_gaussian

TI = training_image("WalkerLake")
truth = training_image("WalkerLakeTruth")

G(m) = imfilter_gaussian(m, [10,10,0])

data = SoftData(G(truth), G)

reals = iqsim(TI, 27, 27, 1, size(truth)..., soft=data, nreal=3)
```
