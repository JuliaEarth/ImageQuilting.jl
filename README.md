[![Build Status](https://travis-ci.org/juliohm/ImageQuilting.jl.png)](https://travis-ci.org/juliohm/ImageQuilting.jl)

ImageQuilting.jl
================

Image quilting for texture synthesis in Julia.

Installation
------------

```julia
Pkg.add("ImageQuilting")
```

Usage
-----

```julia
synthesis = imquilt(img::Image, tilesize, n; tol=1e-3, show=false)
synthesis = imquilt(img::AbstractArray, tilesize, n; tol=1e-3, show=false)
```

where:

* `img` can be any 2D (RGB or Grayscale) image
* `tilesize` is the tile size used to scan `img`
* `n` is the number of tiles to stitch together in the output
* `tol` is the tolerance used for finding best tiles
* `show` tells whether to show the output image or not

Example
-------

Install [ImageView](https://github.com/timholy/ImageView.jl) and
reproduce some of the paper results with:

```julia
ImageQuilting.example()
```

REFERENCES
----------

Efros, A.; Freeman, W. T., 2001. Image Quilting for Texture Synthesis and Transfer. [[DOWNLOAD](http://graphics.cs.cmu.edu/people/efros/research/quilting.html)]
