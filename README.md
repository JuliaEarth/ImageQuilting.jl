[![Build Status](https://travis-ci.org/juliohm/ImageQuilting.jl.svg?branch=master)](https://travis-ci.org/juliohm/ImageQuilting.jl)
[![ImageQuilting](http://pkg.julialang.org/badges/ImageQuilting_nightly.svg)](http://pkg.julialang.org/?pkg=ImageQuilting&ver=nightly)

ImageQuilting.jl
================

3D image quilting simulation.

Installation
------------

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
              seed=0, nreal=1, cutoff=1e-2, categorical=false,
              soft=nothing, debug=false)
```

where:

[required]

* `training_image` can be any 3D array (add ghost dimension for 2D)
* `tplsizex`,`tplsizey`,`tplsizez` is the template size
* `gridsizex`,`gridsizey`,`gridsizez` is the simulation size

[optional]

* `overlapx`,`overlapy`,`overlapz` is the percentage of overlap
* `seed` is the random seed
* `nreal` is the number of realizations
* `cutoff` is the overlap cutoff
* `categorical` informs whether the image is categorical or continuous
* `soft` is an instance of `SoftData`
* `debug` tells whether to export or not the boundary cuts and voxel reusage

The main output `reals` consists of a list of 3D realizations that can be indexed with
`reals[1], reals[2], ..., reals[nreal]`. If `debug=true`, additional output is generated:

```julia
reals, cuts, voxs = iqsim(..., debug=true)
```

`cuts[i]` is the boundary cut for `reals[i]` and `voxs[i]` is the associated voxel reusage.

### Soft data

Given 3D `data` at least as large as the simulation size and a `transform` such that
`transform(training_image)` is comparable with `data`, the `SoftData(data, transform)`
instance can be passed to `iqsim` for local relaxation:

```julia
iqsim(..., soft=SoftData(seismic, blur))
```

Example
-------

Consider installing the [GeoStatsImages.jl](https://github.com/juliohm/GeoStatsImages.jl) package.

```julia
using ImageQuilting
using GeoStatsImages

TI = training_image("Strebelle")
reals = @time iqsim(TI, 62, 62, 1, size(TI)..., nreal=3)

TI = training_image("StoneWall")
reals, cuts, voxs = @time iqsim(TI, 13, 13, 1, size(TI)..., nreal=3, debug=true)
```

REFERENCES
----------

Efros, A.; Freeman, W. T., 2001. Image Quilting for Texture Synthesis and Transfer. [[DOWNLOAD](http://graphics.cs.cmu.edu/people/efros/research/quilting.html)]

Mahmud, K.; Mariethoz, G.; Caers, J.; Tahmasebi, P.; Baker, A., 2014. Simulation of Earth textures by conditional image quilting. [[DOWNLOAD](http://dx.doi.org/10.1002/2013WR015069)]
