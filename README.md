ImageQuilting.jl
================

Fast 3D image quilting simulation.

[![Build Status](https://travis-ci.org/juliohm/ImageQuilting.jl.svg?branch=master)](https://travis-ci.org/juliohm/ImageQuilting.jl)
[![ImageQuilting](http://pkg.julialang.org/badges/ImageQuilting_0.4.svg)](http://pkg.julialang.org/?pkg=ImageQuilting&ver=0.4)
[![Coverage Status](https://coveralls.io/repos/juliohm/ImageQuilting.jl/badge.svg?branch=master)](https://coveralls.io/r/juliohm/ImageQuilting.jl?branch=master)

![3D Quilting Animation](quilting.gif)

Features
--------

* Soft data conditioning
* Hard data conditioning
* Masked grids

Installation
------------

For the latest stable release:

```julia
Pkg.add("ImageQuilting")
```

Usage
-----

```julia
reals = iqsim(training_image::AbstractArray,
              tplsizex::Integer, tplsizey::Integer, tplsizez::Integer,
              gridsizex::Integer, gridsizey::Integer, gridsizez::Integer;
              overlapx=1/6, overlapy=1/6, overlapz=1/6,
              soft=nothing, hard=nothing, cutoff=.1,
              seed=0, nreal=1, categorical=false, debug=false)
```

where:

#### required

* `training_image` can be any 3D array (add ghost dimension for 2D)
* `tplsizex`,`tplsizey`,`tplsizez` is the template size
* `gridsizex`,`gridsizey`,`gridsizez` is the simulation size

#### optional

* `overlapx`,`overlapy`,`overlapz` is the percentage of overlap
* `soft` is an instance of `SoftData` or an array of such instances
* `hard` is an instance of `HardData`
* `cutoff` is the overlap cutoff
* `seed` is the random seed
* `nreal` is the number of realizations
* `categorical` informs whether the image is categorical or continuous
* `debug` tells whether to export or not the boundary cuts and voxel reuse

The main output `reals` consists of a list of 3D realizations that can be indexed with
`reals[1], reals[2], ..., reals[nreal]`. If `debug=true`, additional output is generated:

```julia
reals, cuts, voxs = iqsim(..., debug=true)
```

`cuts[i]` is the boundary cut for `reals[i]` and `voxs[i]` is the associated voxel reuse.

A helper function is also provided for the fast approximation of the *mean voxel reuse*:

```julia
mvr = meanvoxreuse(training_image::AbstractArray,
                   tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;
                   overlapx=1/6, overlapy=1/6, overlapz=1/6,
                   nreal=10, categorical=false)
```

with `mvr` in the interval [0,1]. The approximation gets better as `nreal` is made larger.

### Soft data

Given 3D `data` at least as large as the simulation size and a `transform` such that
`transform(training_image)` is comparable with `data`, the `SoftData(data, transform)`
instance can be passed to `iqsim` for local relaxation:

```julia
iqsim(..., soft=SoftData(seismic, blur))
```

### Hard data

Voxels can be assigned values that will be honored by the simulation. `HardData()` is a dictionary of locations and associated values specified by the user:

```julia
well = HardData([(i,j,k)=>value(i,j,k) for i=10, j=10, k=1:100])
iqsim(..., hard=well)
```

### Masked grids

Masked grids are a special case of hard data conditioning where inactive voxels are marked with the value `NaN`. The algorithm handles this hard data differently as it shouldn't be considered in the pattern similarity calculations.

`training_image` can also have inactive voxels marked with `NaN`. Convolution results are only looked up in active regions.

Example
-------

Consider installing the [GeoStatsImages.jl](https://github.com/juliohm/GeoStatsImages.jl) package.

```julia
using ImageQuilting
using GeoStatsImages

TI = training_image("Strebelle")
reals = iqsim(TI, 62, 62, 1, size(TI)..., nreal=3)

TI = training_image("StoneWall")
reals, cuts, voxs = iqsim(TI, 13, 13, 1, size(TI)..., nreal=3, debug=true)
```

REFERENCES
----------

Efros, A.; Freeman, W. T., 2001. Image Quilting for Texture Synthesis and Transfer. [[DOWNLOAD](http://graphics.cs.cmu.edu/people/efros/research/quilting.html)]

Kwatra, V.; Schodl, A.; Essa, I.; Turk, G.; Bobick, A., 2003. Graphcut Textures: Image and Video Synthesis using Graph Cuts. [[DOWNLOAD](http://www.cc.gatech.edu/~turk/my_papers/graph_cuts.pdf)]

Efros, A.; Leung, T., 1999. Texture Synthesis by Non-parametric Sampling. [[DOWNLOAD](http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html)]
