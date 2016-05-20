ImageQuilting.jl
================

A Julia package for fast 3D image quilting simulation.

[![Build Status](https://travis-ci.org/juliohm/ImageQuilting.jl.svg?branch=master)](https://travis-ci.org/juliohm/ImageQuilting.jl)
[![ImageQuilting](http://pkg.julialang.org/badges/ImageQuilting_0.4.svg)](http://pkg.julialang.org/?pkg=ImageQuilting&ver=0.4)
[![Coverage Status](https://coveralls.io/repos/juliohm/ImageQuilting.jl/badge.svg?branch=master)](https://coveralls.io/r/juliohm/ImageQuilting.jl?branch=master)

![3D Quilting Animation](docs/src/images/quilting.gif)

Features
--------

* Masked grids
* Hard data conditioning
* Soft data conditioning
* Fast computation with GPUs

Installation
------------

Get the latest stable release with Julia's package manager:

```julia
Pkg.add("ImageQuilting")
```

Documentation
-------------

Please refer to the official documentation:

[![Latest Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliohm.github.io/ImageQuilting.jl/latest)

REFERENCES
----------

Efros, A.; Freeman, W. T., 2001. Image Quilting for Texture Synthesis and Transfer. [[DOWNLOAD](http://graphics.cs.cmu.edu/people/efros/research/quilting.html)]

Kwatra, V.; Schodl, A.; Essa, I.; Turk, G.; Bobick, A., 2003. Graphcut Textures: Image and Video Synthesis using Graph Cuts. [[DOWNLOAD](http://www.cc.gatech.edu/~turk/my_papers/graph_cuts.pdf)]

Crimisini, P. P. A; Toyama, K., 2003. Object Removal by Exemplar-Based Inpainting. [[DOWNLOAD](http://research.microsoft.com/pubs/67273/criminisi_cvpr2003.pdf)]
