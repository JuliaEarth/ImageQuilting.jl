Below are the concepts implemented in this package. For understanding how these concepts
are used, please consult the [Examples](examples.md) section.

## Soft data

Given 3D `data` at least as large as the simulation size and a `transform` such that
`transform(training_image)` is comparable with `data`, the `SoftData(data, transform)`
instance can be passed to `iqsim` for local relaxation:

```julia
iqsim(..., soft=SoftData(seismic, blur))
```

## Hard data

Voxels can be assigned values that will be honored by the simulation. `HardData()` is
a dictionary of locations and associated values specified by the user:

```julia
well = HardData((i,j,k)=>value(i,j,k) for i=10, j=10, k=1:100)
iqsim(..., hard=well)
```

## Masked grids

Masked grids are a special case of hard data conditioning where inactive voxels are
marked with the value `NaN`. The algorithm handles this hard data differently as it
shouldn't be considered in the pattern similarity calculations.

`training_image` can also have inactive voxels marked with `NaN`. Convolution results
are only looked up in active regions.
