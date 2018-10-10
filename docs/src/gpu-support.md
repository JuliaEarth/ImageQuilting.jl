!!! note "Disclaimer"

    GPGPU is one of the most unportable corners in the programming world. Although I did make use
    of [OpenCL](https://www.khronos.org/opencl) in this package, drivers for graphics cards are
    problematic and vendors such as NVIDIA do not officially support widely known operating systems.

Two external dependencies need to be manually installed:

* OpenCL driver
* clFFT C++ library

## Installing OpenCL driver

The choice of the OpenCL driver is dependent on the graphics card you have. Find what is the model
of your graphics card and download the appropriate driver from the links below:

* [Intel](https://software.intel.com/en-us/articles/opencl-drivers)
* [AMD](http://support.amd.com/en-us/download)
* [NVIDIA](http://www.nvidia.com/Download/index.aspx)

If you are on Linux like myself, check the repositories of your distribution for a more straightforward
installation. If you have a recent Intel graphics card, consider the open source
[Beignet driver](https://www.freedesktop.org/wiki/Software/Beignet) also available in some distributions
(e.g. AUR repos in Arch Linux).

To make sure that everything is working properly, install the `OpenCL.jl` package in Julia and run the tests:

```julia
] add OpenCL
] test OpenCL
```

If the tests are successful, proceed to the next section.

## Installing clFFT C++ library

Download and install the [pre-built binaries](https://github.com/clMathLibraries/clFFT/releases). If you
are on Linux, you can also check the repositories of your distribution.

Install the `CLFFT.jl` package in Julia and run the tests:

```julia
] add CLFFT
] test CLFFT
```

If the tests are successful, the installation is complete.

## Testing GPU implementation

Run the tests to make sure that the GPU implementation is working as expected:

```julia
] test ImageQuilting
```

Pass in the option `gpu=true` to `iqsim` for computations with the GPU.
