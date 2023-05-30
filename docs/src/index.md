## Overview

*A Julia package for fast 3D image quilting simulation.*

[![Build Status](https://img.shields.io/github/actions/workflow/status/JuliaEarth/ImageQuilting.jl/CI.yml?branch=master&style=flat-square)](https://github.com/JuliaEarth/ImageQuilting.jl/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/JuliaEarth/ImageQuilting.jl)](https://codecov.io/gh/JuliaEarth/ImageQuilting.jl)
[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaEarth.github.io/ImageQuilting.jl/stable)
[![Latest Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaEarth.github.io/ImageQuilting.jl/dev)

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
] add ImageQuilting
```

## Talks

Below is a list of talks related to this project. For more material, please subscribe to the
[YouTube channel](https://www.youtube.com/channel/UCiOnsyYAZM-voi5diu8lN9w).

```@raw html
<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/YJs7jl_Y9yM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>
<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Y5KhQCapuPw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>
```

## Usage

This package is part of the [GeoStats.jl](https://github.com/JuliaEarth/GeoStats.jl) framework.
Solver options are displayed below:

```@docs
IQ
```

### Low-level API

If you are interested in using the package without GeoStats.jl, please use the following function:

```@docs
iqsim
```

The major difference compared to the high-level API is that the `iqsim` function has
no notion of coordinate system, and you will have to pre/post-process the data manually
to match it with the cells in the simulation grid.

GeoStats.jl takes the coordinate system into account and also enables parallel simulation
on clusters of computers with distributed memory.
