Below are the concepts implemented in this package. For understanding how these concepts
are used, please consult the [Examples](examples.md) section.

## Hard data

Voxels can be assigned values that will be honored by the simulation. `HardData()` is
a dictionary of locations and associated values specified by the user:

```julia
well = HardData((i,j,k)=>value(i,j,k) for i=10, j=10, k=1:100)
iqsim(..., hard=well)
```

## Soft data

Given 3D `data` of size `(gridsizex, gridsizey, gridsizez)` and `dataTI` of size
`size(training_image)`, local relaxation can be performed with:

```julia
# 3D seismic as auxiliary data
iqsim(..., soft=[(seismic,seismicTI)])
```

Multiple pairs of data can be passed as well:

```julia
iqsim(..., soft=[(data1,dataTI1), (data2,dataTI2), ...])
```

## Masked grids

Masked grids are a special case of hard data conditioning where inactive voxels are
marked with the value `NaN`. The algorithm handles this hard data differently as it
shouldn't be considered in the pattern similarity calculations.

`training_image` can also have inactive voxels marked with `NaN`. Convolution results
are only looked up in active regions.
