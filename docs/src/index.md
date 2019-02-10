## Overview

*A Julia package for fast 3D image quilting simulation.*

[![Build Status](https://travis-ci.org/juliohm/ImageQuilting.jl.svg?branch=master)](https://travis-ci.org/juliohm/ImageQuilting.jl)
[![Coverage Status](https://coveralls.io/repos/juliohm/ImageQuilting.jl/badge.svg?branch=master)](https://coveralls.io/r/juliohm/ImageQuilting.jl?branch=master)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliohm.github.io/ImageQuilting.jl/stable)
[![Latest Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliohm.github.io/ImageQuilting.jl/latest)

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

This package is part of the [GeoStats.jl](https://github.com/juliohm/GeoStats.jl) framework. Solver
options are displayed below:

```@docs
ImgQuilt
```

### Low-level API

If you are interested in using the package without GeoStats.jl, please use the following function:

```@docs
iqsim
```

The major difference compared to the high-level API is that the `iqsim` function has
no notion of coordinate system, and you will have to pre-process the data manually to
match it with the cells in the simulation grid.

GeoStats.jl takes the coordinate system into account and also enables parallel simulation
on HPC clusters.
