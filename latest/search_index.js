var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Overview-1",
    "page": "Home",
    "title": "Overview",
    "category": "section",
    "text": "A Julia package for fast 3D image quilting simulation.(Image: Build Status) (Image: ImageQuilting) (Image: Coverage Status) (Image: Stable Documentation) (Image: Latest Documentation)This package implements an extension to the famous Efros-Freeman algorithm for texture synthesis and transfer in computer vision. Unlike the original algorithm developed for 2D images, our method can also handle 3D masked grids and pre-existing point-data very efficiently (the fastest in the literature). For more details, please refer to our paper in Citation.(Image: 3D Quilting Animation)"
},

{
    "location": "index.html#Features-1",
    "page": "Home",
    "title": "Features",
    "category": "section",
    "text": "Masked grids\nHard data conditioning\nSoft data conditioning\nFast computation with GPUs"
},

{
    "location": "index.html#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "Get the latest stable release with Julia\'s package manager:Pkg.add(\"ImageQuilting\")For even faster computation with GPUs, please follow the instructions in GPU support."
},

{
    "location": "index.html#ImageQuilting.ImgQuilt",
    "page": "Home",
    "title": "ImageQuilting.ImgQuilt",
    "category": "type",
    "text": "ImgQuilt(var₁=>param₁, var₂=>param₂, ...)\n\nImage quilting simulation solver as described in Hoffimann et al. 2017.\n\nParameters\n\nRequired\n\nTI       - Training image\ntilesize - Tile size in x, y and z\n\nOptional\n\noverlap  - Overlap size in x, y and z (default to (1/6, 1/6, 1/6))\ncut      - Boundary cut algorithm (:boykov (default) or :dijkstra)\npath     - Simulation path (:rasterup (default), :rasterdown, :dilation, or :random)\ninactive - Vector of inactive voxels (i.e. tuples (i,j,k)) in the grid\nsoft     - A vector of (data,dataTI) pairs\ntol      - Initial relaxation tolerance in (0,1] (default to 0.1)\n\nGlobal parameters\n\nOptional\n\nthreads      - Number of threads in FFT (default to number of physical CPU cores)\ngpu          - Whether to use the GPU or the CPU (default to false)\nshowprogress - Whether to show or not the estimated time duration (default to false)\n\n\n\n\n\n\n\n"
},

{
    "location": "index.html#Usage-1",
    "page": "Home",
    "title": "Usage",
    "category": "section",
    "text": "This package is part of the GeoStats.jl framework. Solver options are displayed below:ImgQuilt"
},

{
    "location": "index.html#ImageQuilting.iqsim",
    "page": "Home",
    "title": "ImageQuilting.iqsim",
    "category": "function",
    "text": "iqsim(trainimg::AbstractArray{T,N},\n      tilesize::Dims{N}, gridsize::Dims{N};\n      overlap::NTuple{N,Float64}=ntuple(i->1/6,N),\n      soft::AbstractVector=[], hard::HardData=HardData(), tol::Real=.1,\n      cut::Symbol=:boykov, path::Symbol=:rasterup, nreal::Integer=1,\n      threads::Integer=cpucores(), gpu::Bool=false,\n      debug::Bool=false, showprogress::Bool=false)\n\nPerforms image quilting simulation as described in Hoffimann et al. 2017.\n\nParameters\n\nRequired\n\ntrainimg is any 3D array (add ghost dimension for 2D)\ntilesize is the tile size (or pattern size)\ngridsize is the size of the simulation grid\n\nOptional\n\noverlap is the percentage of overlap\nsoft is a vector of (data,dataTI) pairs\nhard is an instance of HardData\ntol is the initial relaxation tolerance in (0,1] (default to .1)\ncut is the cut algorithm (:dijkstra or :boykov)\npath is the simulation path (:rasterup, :rasterdown, :dilation or :random)\nnreal is the number of realizations\nthreads is the number of threads for the FFT (default to all CPU cores)\ngpu informs whether to use the GPU or the CPU\ndebug informs whether to export or not the boundary cuts and voxel reuse\nshowprogress informs whether to show or not estimated time duration\n\nThe main output reals consists of a list of 3D realizations that can be indexed with reals[1], reals[2], ..., reals[nreal]. If debug=true, additional output is generated:\n\nreals, cuts, voxs = iqsim(..., debug=true)\n\ncuts[i] is the boundary cut for reals[i] and voxs[i] is the associated voxel reuse.\n\n\n\n\n\n"
},

{
    "location": "index.html#Low-level-API-1",
    "page": "Home",
    "title": "Low-level API",
    "category": "section",
    "text": "If you are interested in using the package without GeoStats.jl, please use the following function:iqsimThe major difference compared to the high-level API is that the iqsim function has no notion of coordinate system, and you will have to pre-process the data manually to match it with the cells in the simulation grid.GeoStats.jl takes the coordinate system into account and also enables parallel simulation on HPC clusters."
},

{
    "location": "concepts.html#",
    "page": "Concepts",
    "title": "Concepts",
    "category": "page",
    "text": "Below are the concepts implemented in this package. For understanding how these concepts are used, please consult the Examples section."
},

{
    "location": "concepts.html#Hard-data-1",
    "page": "Concepts",
    "title": "Hard data",
    "category": "section",
    "text": "Voxels can be assigned values that will be honored by the simulation. HardData() is a dictionary of locations and associated values specified by the user:well = HardData(CartesianIndex(i,j,k)=>value(i,j,k) for i=10, j=10, k=1:100)\niqsim(..., hard=well)"
},

{
    "location": "concepts.html#Soft-data-1",
    "page": "Concepts",
    "title": "Soft data",
    "category": "section",
    "text": "Given 3D data of the same size of the simulation grid (e.g. seismic) and data of the same of the training image (e.g. seismicTI), local relaxation can be performed with:# 3D seismic as auxiliary data\niqsim(..., soft=[(seismic,seismicTI)])Multiple pairs of data can be passed as well:iqsim(..., soft=[(data₁,dataTI₁), (data₂,dataTI₂), ...])"
},

{
    "location": "concepts.html#Masked-grids-1",
    "page": "Concepts",
    "title": "Masked grids",
    "category": "section",
    "text": "Masked grids are a special case of hard data conditioning where inactive voxels are marked with the value NaN. The algorithm handles this hard data differently as it shouldn\'t be considered in the pattern similarity calculations.The training image can also have inactive voxels marked with NaN. Convolution results are only looked up in active regions."
},

{
    "location": "examples.html#",
    "page": "Examples",
    "title": "Examples",
    "category": "page",
    "text": "note: Note\nConsider installing the GeoStatsImages.jl package."
},

{
    "location": "examples.html#Unconditional-1",
    "page": "Examples",
    "title": "Unconditional",
    "category": "section",
    "text": "An example of unconditional simulation (i.e. simulation without data). This is the original Efros-Freeman algorithm for texture synthesis with a few additional options.using ImageQuilting\nusing GeoStatsImages\n\nTI = training_image(\"Strebelle\")\nreals = iqsim(TI, (62, 62, 1), size(TI), nreal=3)\n\nTI = training_image(\"StoneWall\")\nreals, cuts, voxs = iqsim(TI, (13, 13, 1), size(TI), nreal=3, debug=true)(Image: Unconditional simulation)"
},

{
    "location": "examples.html#Hard-data-1",
    "page": "Examples",
    "title": "Hard data",
    "category": "section",
    "text": "Hard data (i.e. point data) can be honored during simulation. This can be useful when images represent a spatial property of a physical domain. For example, the subsurface of the Earth is only known with certainty at a few well locations.using ImageQuilting\nusing GeoStatsImages\n\nTI = training_image(\"Strebelle\")\n\ndata = HardData()\npush!(data, CartesianIndex(50,50,1)=>1)\npush!(data, CartesianIndex(190,50,1)=>0)\npush!(data, CartesianIndex(150,170,1)=>1)\npush!(data, CartesianIndex(150,190,1)=>1)\n\nreals, cuts, voxs = iqsim(TI, (30, 30, 1), size(TI), hard=data, debug=true)(Image: Hard data conditioning)(Image: Hard data conditioning)"
},

{
    "location": "examples.html#Soft-data-1",
    "page": "Examples",
    "title": "Soft data",
    "category": "section",
    "text": "Sometimes it is also useful to incorporate auxiliary variables defined in the domain, which can guide the selection of patterns in the training image. This example shows how to achieve this texture transfer efficiently.using ImageQuilting\nusing GeoStatsImages\nusing Images\n\nTI    = training_image(\"WalkerLake\")\ntruth = training_image(\"WalkerLakeTruth\")\n\nG(m) = imfilter(m, KernelFactors.IIRGaussian([10,10,0]))\n\ndata   = G(truth)\ndataTI = G(TI)\n\nreals = iqsim(TI, (27, 27, 1), size(truth), soft=[(data,dataTI)], nreal=3)(Image: Soft data conditioning)"
},

{
    "location": "examples.html#Masked-grids-1",
    "page": "Examples",
    "title": "Masked grids",
    "category": "section",
    "text": "Voxels marked with the special symbol NaN are treated as inactive. The algorithm will skip tiles that only contain inactive voxels to save computation and will generate realizations that are consistent with the mask. This is particularly useful with complex 3D models that have large inactive portions.using ImageQuilting\nusing GeoStatsImages\n\nTI = training_image(\"Strebelle\")\nnx, ny = size(TI)\n\n# skip circle at the center\nr = 100; shape = HardData()\nfor i=1:size(TI, 1), j=1:size(TI, 2)\n    if (i-nx÷2)^2 + (j-ny÷2)^2 < radius^2\n        push!(shape, CartesianIndex(i,j,1)=>NaN)\n    end\nend\n\nreals = iqsim(TI, (62, 62, 1), size(TI), hard=shape, nreal=3)(Image: Masked grids)"
},

{
    "location": "voxel-reuse.html#",
    "page": "Voxel reuse",
    "title": "Voxel reuse",
    "category": "page",
    "text": ""
},

{
    "location": "voxel-reuse.html#ImageQuilting.voxelreuse",
    "page": "Voxel reuse",
    "title": "ImageQuilting.voxelreuse",
    "category": "function",
    "text": "voxelreuse(trainimg::AbstractArray{T,N}, tilesize::Dims{N};\n           overlap::NTuple{N,Float64}=ntuple(i->1/6,N),\n           nreal::Integer=10, kwargs...)\n\nReturns the mean voxel reuse in [0,1] and its standard deviation.\n\nNotes\n\nThe approximation gets better as nreal is made larger.\nKeyword arguments kwargs are passed to iqsim directly.\n\n\n\n\n\n"
},

{
    "location": "voxel-reuse.html#Helper-function-1",
    "page": "Voxel reuse",
    "title": "Helper function",
    "category": "section",
    "text": "A helper function is provided for the fast approximation of the mean voxel reuse:voxelreuse"
},

{
    "location": "voxel-reuse.html#Plot-recipe-1",
    "page": "Voxel reuse",
    "title": "Plot recipe",
    "category": "section",
    "text": "A plot recipe is provided for tile design in image quilting. In order to plot the voxel reuse of a training image, install Plots.jl and any of its supported backends (e.g. GR.jl):] add Plots GRThe example below uses training images from the GeoStatsImages.jl package:using ImageQuilting\nusing GeoStatsImages\nusing Plots\n\nTI₁ = training_image(\"Strebelle\")\nTI₂ = training_image(\"StoneWall\")\n\nvoxelreuseplot(TI₁, label=\"Strebelle\")\nvoxelreuseplot!(TI₂, label=\"StoneWall\")(Image: Voxel reuse plot)"
},

{
    "location": "gpu-support.html#",
    "page": "GPU support",
    "title": "GPU support",
    "category": "page",
    "text": "note: Disclaimer\nGPGPU is one of the most unportable corners in the programming world. Although I did make use of OpenCL in this package, drivers for graphics cards are problematic and vendors such as NVIDIA do not officially support widely known operating systems.Two external dependencies need to be manually installed:OpenCL driver\nclFFT C++ library"
},

{
    "location": "gpu-support.html#Installing-OpenCL-driver-1",
    "page": "GPU support",
    "title": "Installing OpenCL driver",
    "category": "section",
    "text": "The choice of the OpenCL driver is dependent on the graphics card you have. Find what is the model of your graphics card and download the appropriate driver from the links below:Intel\nAMD\nNVIDIAIf you are on Linux like myself, check the repositories of your distribution for a more straightforward installation. If you have a recent Intel graphics card, consider the open source Beignet driver also available in some distributions (e.g. AUR repos in Arch Linux).To make sure that everything is working properly, install the OpenCL.jl package in Julia and run the tests:] add OpenCL\n] test OpenCLIf the tests are successful, proceed to the next section."
},

{
    "location": "gpu-support.html#Installing-clFFT-C-library-1",
    "page": "GPU support",
    "title": "Installing clFFT C++ library",
    "category": "section",
    "text": "Download and install the pre-built binaries. If you are on Linux, you can also check the repositories of your distribution.Install the CLFFT.jl package in Julia and run the tests:] add CLFFT\n] test CLFFTIf the tests are successful, the installation is complete."
},

{
    "location": "gpu-support.html#Testing-GPU-implementation-1",
    "page": "GPU support",
    "title": "Testing GPU implementation",
    "category": "section",
    "text": "Run the tests to make sure that the GPU implementation is working as expected:] test ImageQuiltingPass in the option gpu=true to iqsim for computations with the GPU."
},

{
    "location": "about/author.html#",
    "page": "Author",
    "title": "Author",
    "category": "page",
    "text": "Júlio Hoffimann MendesI am a Ph.D. candidate in the Department of Energy Resources Engineering at Stanford University. You can find more about my research on my website. Below are some ways that we can connect:ResearchGate\nLinkedIn\nGitHub"
},

{
    "location": "about/license.html#",
    "page": "License",
    "title": "License",
    "category": "page",
    "text": "The ImageQuilting.jl package is licensed under the ISC License:Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>\n\nPermission to use, copy, modify, and/or distribute this software for any\npurpose with or without fee is hereby granted, provided that the above\ncopyright notice and this permission notice appear in all copies.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\" AND THE AUTHOR DISCLAIMS ALL WARRANTIES\nWITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF\nMERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR\nANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES\nWHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN\nACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF\nOR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE."
},

{
    "location": "about/citation.html#",
    "page": "Citation",
    "title": "Citation",
    "category": "page",
    "text": "If you find ImageQuilting.jl useful in your work, please consider citing our paper:@ARTICLE{Hoffimann2017,\n  title={Stochastic Simulation by Image Quilting of Process-based Geological Models},\n  author={Hoffimann, J{\\\'u}lio and Scheidt, C{\\\'e}line and Barfod, Adrian and Caers, Jef},\n  journal={Computers \\& Geosciences},\n  publisher={Elsevier BV},\n  volume={106},\n  pages={18-32},\n  ISSN={0098-3004},\n  DOI={10.1016/j.cageo.2017.05.012},\n  url={http://dx.doi.org/10.1016/j.cageo.2017.05.012},\n  year={2017},\n  month={May}\n}"
},

]}
