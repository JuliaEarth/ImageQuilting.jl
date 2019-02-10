ImageQuilting.jl
================

*A Julia package for fast 3D image quilting simulation.*

[![][travis-img]][travis-url] [![][codecov-img]][codecov-url] [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

![3D Quilting Animation](docs/src/images/quilting.gif)

## Latest News
- ImageQuilting.jl won the [Syvitski Modeler Award 2018](https://csdms.colorado.edu/wiki/Student_Modeler_Award_2018).
- ImageQuilting.jl can now be used as one of the many solvers in the
[GeoStats.jl](https://github.com/juliohm/GeoStats.jl) framework.
For more information, please type `?ImgQuilt` in the Julia prompt
after loading the package.

Installation
------------

Get the latest stable release with Julia's package manager:

```julia
] add ImageQuilting
```

Documentation
-------------

- [**STABLE**][docs-stable-url] &mdash; **most recently tagged version of the documentation.**
- [**LATEST**][docs-latest-url] &mdash; *in-development version of the documentation.*

Citation
--------

If you find ImageQuilting.jl useful in your work, please consider citing our paper:

```latex
@ARTICLE{Hoffimann2017,
  title={Stochastic Simulation by Image Quilting of Process-based Geological Models},
  author={Hoffimann, J{\'u}lio and Scheidt, C{\'e}line and Barfod, Adrian and Caers, Jef},
  journal={Computers \& Geosciences},
  publisher={Elsevier BV},
  volume={106},
  pages={18-32},
  ISSN={0098-3004},
  DOI={10.1016/j.cageo.2017.05.012},
  url={http://dx.doi.org/10.1016/j.cageo.2017.05.012},
  year={2017},
  month={May}
}
```

Publications
------------

- Barfod et al. 2017. *Hydrostratigraphic modelling using multiple-point statistics and airborne transient electromagnetic methods* [DOWNLOAD](https://www.researchgate.net/publication/319235285_Hydrostratigraphic_modelling_using_multiple-point_statistics_and_airborne_transient_electromagnetic_methods)

- Hoffimann et al. 2017. *Stochastic Simulation by Image Quilting of Process-based Geological Models*
[DOWNLOAD](https://www.researchgate.net/publication/317151543_Stochastic_Simulation_by_Image_Quilting_of_Process-based_Geological_Models)

- Hoffimann et al. 2015. *Geostatistical Modeling of Evolving Landscapes by Means of Image Quilting*
[DOWNLOAD](https://www.researchgate.net/publication/295902985_Geostatistical_Modeling_of_Evolving_Landscapes_by_Means_of_Image_Quilting)

Talks
-----

#### CSDMS 2018
[![CSDMS2018](https://img.youtube.com/vi/Y5KhQCapuPw/0.jpg)](https://www.youtube.com/watch?v=Y5KhQCapuPw)

#### JuliaCon 2017
[![JuliaCon2017](https://img.youtube.com/vi/YJs7jl_Y9yM/0.jpg)](https://www.youtube.com/watch?v=YJs7jl_Y9yM)

Contributing
------------

Contributions are very welcome, as are feature requests and suggestions.

Please [open an issue](https://github.com/juliohm/ImageQuilting.jl/issues) if you encounter any problems.

[travis-img]: https://travis-ci.org/juliohm/ImageQuilting.jl.svg?branch=master
[travis-url]: https://travis-ci.org/juliohm/ImageQuilting.jl

[codecov-img]: https://codecov.io/gh/juliohm/ImageQuilting.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/juliohm/ImageQuilting.jl

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://juliohm.github.io/ImageQuilting.jl/stable

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://juliohm.github.io/ImageQuilting.jl/latest
