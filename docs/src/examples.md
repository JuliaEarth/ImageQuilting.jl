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

TODO
