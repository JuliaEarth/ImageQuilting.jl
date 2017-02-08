## Overview

A Julia package for fast 3D image quilting simulation.

[![Build Status](https://travis-ci.org/juliohm/ImageQuilting.jl.svg?branch=master)](https://travis-ci.org/juliohm/ImageQuilting.jl)
[![ImageQuilting](http://pkg.julialang.org/badges/ImageQuilting_0.5.svg)](http://pkg.julialang.org/?pkg=ImageQuilting)
[![Coverage Status](https://coveralls.io/repos/juliohm/ImageQuilting.jl/badge.svg?branch=master)](https://coveralls.io/r/juliohm/ImageQuilting.jl?branch=master)

This package implements an extension to the famous Efros-Freeman algorithm for texture synthesis and transfer in computer vision.
Unlike the original algorithm developed for 2D images, our method can also handle 3D masked grids and pre-existing point-data very
efficiently (the fastest in the literature). For more details, please refer to our paper in [Citation](about/citation.md).

![3D Quilting Animation](images/quilting.gif)

## Features

- Masked grids
- Hard data conditioning
- Soft data conditioning
- Fast computation with GPUs

## Installation

Get the latest stable release with Julia's package manager:

```julia
Pkg.add("ImageQuilting")
```

For even faster computation with GPUs, please follow the instructions in [GPU support](gpu-support.md).

## Usage

```julia
reals = iqsim(training_image::AbstractArray,
              tplsizex::Integer, tplsizey::Integer, tplsizez::Integer,
              gridsizex::Integer, gridsizey::Integer, gridsizez::Integer;
              overlapx=1/6, overlapy=1/6, overlapz=1/6,
              soft=nothing, hard=nothing, tol=.1,
              cut=:boykov, path=:rasterup, simplex=false, nreal=1,
              threads=CPU_PHYSICAL_CORES, gpu=false, debug=false, showprogress=false)
```

where:

**required**

- `training_image` can be any 3D array (add ghost dimension for 2D)
- `tplsizex`,`tplsizey`,`tplsizez` is the template size
- `gridsizex`,`gridsizey`,`gridsizez` is the simulation size

**optional**

- `overlapx`,`overlapy`,`overlapz` is the percentage of overlap
- `soft` is an instance of `SoftData` or an array of such instances
- `hard` is an instance of `HardData`
- `tol` is the initial relaxation tolerance in (0,1]
- `cut` is the cut algorithm (`:dijkstra` or `:boykov`)
- `path` is the simulation path (`:rasterup`, `:rasterdown`, `:dilation` or `:random`)
- `simplex` informs whether to apply or not the simplex transform to the image
- `nreal` is the number of realizations
- `threads` is the number of threads for the FFT (default to all CPU cores)
- `gpu` informs whether to use the GPU or the CPU
- `debug` informs whether to export or not the boundary cuts and voxel reuse
- `showprogress` informs whether to show or not estimated time duration

The main output `reals` consists of a list of 3D realizations that can be indexed with
`reals[1], reals[2], ..., reals[nreal]`. If `debug=true`, additional output is generated:

```julia
reals, cuts, voxs = iqsim(..., debug=true)
```

`cuts[i]` is the boundary cut for `reals[i]` and `voxs[i]` is the associated voxel reuse.

A helper function is also provided for the fast approximation of the *mean voxel reuse*:

```julia
mean, dev = voxelreuse(training_image::AbstractArray,
                       tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;
                       overlapx=1/6, overlapy=1/6, overlapz=1/6,
                       cut=:boykov, simplex=false, nreal=10,
                       threads=CPU_PHYSICAL_CORES, gpu=false)
```

with `mean` in the interval ``[0,1]`` and `dev` the standard deviation. The approximation
gets better as `nreal` is made larger.
