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
    "text": "A Julia package for fast 3D image quilting simulation.(Image: Build Status) (Image: ImageQuilting) (Image: Coverage Status)This package implements an extension to the famous Efros-Freeman algorithm for texture synthesis and transfer in computer vision. Unlike the original algorithm developed for 2D images, our method can also handle 3D masked grids and pre-existing point-data very efficiently (the fastest in the literature). For more details, please refer to our paper in Citation.(Image: 3D Quilting Animation)"
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
    "text": "Get the latest stable release with Julia's package manager:Pkg.add(\"ImageQuilting\")For even faster computation with GPUs, please follow the instructions in GPU support."
},

{
    "location": "index.html#Usage-1",
    "page": "Home",
    "title": "Usage",
    "category": "section",
    "text": "reals = iqsim(training_image::AbstractArray,\n              tplsizex::Integer, tplsizey::Integer, tplsizez::Integer,\n              gridsizex::Integer, gridsizey::Integer, gridsizez::Integer;\n              overlapx=1/6, overlapy=1/6, overlapz=1/6,\n              soft=nothing, hard=nothing, tol=.1,\n              cut=:boykov, path=:rasterup, simplex=false, nreal=1,\n              threads=CPU_PHYSICAL_CORES, gpu=false, debug=false, showprogress=false)where:requiredtraining_image can be any 3D array (add ghost dimension for 2D)\ntplsizex,tplsizey,tplsizez is the template size\ngridsizex,gridsizey,gridsizez is the simulation sizeoptionaloverlapx,overlapy,overlapz is the percentage of overlap\nsoft is an instance of SoftData or an array of such instances\nhard is an instance of HardData\ntol is the initial relaxation tolerance in (0,1]\ncut is the cut algorithm (:dijkstra or :boykov)\npath is the simulation path (:rasterup, :rasterdown, :dilation or :random)\nsimplex informs whether to apply or not the simplex transform to the image\nnreal is the number of realizations\nthreads is the number of threads for the FFT (default to all CPU cores)\ngpu informs whether to use the GPU or the CPU\ndebug informs whether to export or not the boundary cuts and voxel reuse\nshowprogress informs whether to show or not estimated time durationThe main output reals consists of a list of 3D realizations that can be indexed with reals[1], reals[2], ..., reals[nreal]. If debug=true, additional output is generated:reals, cuts, voxs = iqsim(..., debug=true)cuts[i] is the boundary cut for reals[i] and voxs[i] is the associated voxel reuse.In addition, this package provides utility functions for template design in image quilting. For more details, please refer to the Voxel reuse section."
},

{
    "location": "concepts.html#",
    "page": "Concepts",
    "title": "Concepts",
    "category": "page",
    "text": "Below are the concepts implemented in this package. For understanding how these concepts are used, please consult the Examples section."
},

{
    "location": "concepts.html#Soft-data-1",
    "page": "Concepts",
    "title": "Soft data",
    "category": "section",
    "text": "Given 3D data at least as large as the simulation size and a transform such that transform(training_image) is comparable with data, the SoftData(data, transform) instance can be passed to iqsim for local relaxation:iqsim(..., soft=SoftData(seismic, blur))"
},

{
    "location": "concepts.html#Hard-data-1",
    "page": "Concepts",
    "title": "Hard data",
    "category": "section",
    "text": "Voxels can be assigned values that will be honored by the simulation. HardData() is a dictionary of locations and associated values specified by the user:well = HardData((i,j,k)=>value(i,j,k) for i=10, j=10, k=1:100)\niqsim(..., hard=well)"
},

{
    "location": "concepts.html#Masked-grids-1",
    "page": "Concepts",
    "title": "Masked grids",
    "category": "section",
    "text": "Masked grids are a special case of hard data conditioning where inactive voxels are marked with the value NaN. The algorithm handles this hard data differently as it shouldn't be considered in the pattern similarity calculations.training_image can also have inactive voxels marked with NaN. Convolution results are only looked up in active regions."
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
    "text": "An example of unconditional simulation (i.e. simulation without data). This is the original Efros-Freeman algorithm for texture synthesis with a few additional options.using ImageQuilting\nusing GeoStatsImages\n\nTI = training_image(\"Strebelle\")\nreals = iqsim(TI, 62, 62, 1, size(TI)..., nreal=3)\n\nTI = training_image(\"StoneWall\")\nreals, cuts, voxs = iqsim(TI, 13, 13, 1, size(TI)..., nreal=3, debug=true)(Image: Unconditional simulation)"
},

{
    "location": "examples.html#Hard-data-1",
    "page": "Examples",
    "title": "Hard data",
    "category": "section",
    "text": "Hard data (i.e. point data) can be honored during simulation. This can be useful when images represent a spatial property of a physical domain. For example, the subsurface of the Earth is only known with certainty at a few well locations.using ImageQuilting\nusing GeoStatsImages\n\nTI = training_image(\"Strebelle\")\n\ndata = HardData()\npush!(data, (50,50,1)=>1)\npush!(data, (190,50,1)=>0)\npush!(data, (150,170,1)=>1)\npush!(data, (150,190,1)=>1)\n\nreals, cuts, voxs = iqsim(TI, 30, 30, 1, size(TI)..., hard=data, debug=true)(Image: Hard data conditioning)(Image: Hard data conditioning)"
},

{
    "location": "examples.html#Soft-data-1",
    "page": "Examples",
    "title": "Soft data",
    "category": "section",
    "text": "Sometimes it is also useful to incorporate auxiliary variables defined in the domain, which can guide the selection of patterns in the training image. This example shows how to achieve this texture transfer efficiently.using ImageQuilting\nusing GeoStatsImages\nusing Images\n\nTI = training_image(\"WalkerLake\")\ntruth = training_image(\"WalkerLakeTruth\")\n\nG(m) = imfilter(m, KernelFactors.IIRGaussian([10,10,0]))\n\ndata = SoftData(G(truth), G)\n\nreals = iqsim(TI, 27, 27, 1, size(truth)..., soft=data, nreal=3)(Image: Soft data conditioning)"
},

{
    "location": "voxel-reuse.html#",
    "page": "Voxel reuse",
    "title": "Voxel reuse",
    "category": "page",
    "text": ""
},

{
    "location": "voxel-reuse.html#Helper-function-1",
    "page": "Voxel reuse",
    "title": "Helper function",
    "category": "section",
    "text": "A helper function is provided for the fast approximation of the mean voxel reuse:mean, dev = voxelreuse(training_image::AbstractArray,\n                       tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;\n                       overlapx=1/6, overlapy=1/6, overlapz=1/6,\n                       cut=:boykov, simplex=false, nreal=10,\n                       threads=CPU_PHYSICAL_CORES, gpu=false)with mean in the interval 01 and dev the standard deviation. The approximation gets better as nreal is made larger."
},

{
    "location": "voxel-reuse.html#Plot-recipe-1",
    "page": "Voxel reuse",
    "title": "Plot recipe",
    "category": "section",
    "text": "A plot recipe is provided for template design in image quilting. In order to plot the voxel reuse of a training image, install Plots.jl and any of its supported backends (e.g. PyPlot.jl):Pkg.add(\"Plots\")\nPkg.add(\"PyPlot\")The example below uses training images from the GeoStatsImages.jl package:using ImageQuilting\nusing GeoStatsImages\nusing Plots\n\nTI₁ = training_image(\"Strebelle\")\nTI₂ = training_image(\"StoneWall\")\n\nplot(VoxelReuse(TI₁), label=\"Strebelle\")\nplot!(VoxelReuse(TI₂), label=\"StoneWall\")(Image: Voxel reuse plot)"
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
    "text": "The choice of the OpenCL driver is dependent on the graphics card you have. Find what is the model of your graphics card and download the appropriate driver from the links below:Intel\nAMD\nNVIDIAIf you are on Linux like myself, check the repositories of your distribution for a more straightforward installation. If you have a recent Intel graphics card, consider the open source Beignet driver also available in some distributions (e.g. AUR repos in Arch Linux).To make sure that everything is working properly, install the OpenCL.jl package in Julia and run the tests:Pkg.add(\"OpenCL\")\n\nusing OpenCL # force compilation\nPkg.test(\"OpenCL\")If the tests are successful, proceed to the next section."
},

{
    "location": "gpu-support.html#Installing-clFFT-C-library-1",
    "page": "GPU support",
    "title": "Installing clFFT C++ library",
    "category": "section",
    "text": "Download and install the pre-built binaries. If you are on Linux, you can also check the repositories of your distribution.Install the CLFFT.jl package in Julia and run the tests:Pkg.add(\"CLFFT\")\n\nusing CLFFT # force compilation\nPkg.test(\"CLFFT\")If the tests are successful, the installation is complete."
},

{
    "location": "gpu-support.html#Testing-GPU-implementation-1",
    "page": "GPU support",
    "title": "Testing GPU implementation",
    "category": "section",
    "text": "Run the tests to make sure that the GPU implementation is working as expected:using ImageQuilting # force compilation\nPkg.test(\"ImageQuilting\")Pass in the option gpu=true to iqsim for computations with the GPU."
},

{
    "location": "about/author.html#",
    "page": "Author",
    "title": "Author",
    "category": "page",
    "text": "Júlio Hoffimann MendesI am a Ph.D. candidate in the Department of Energy Resources Engineering at Stanford University. Below are some ways we can connect:ResearchGate\nLinkedInThis package was inspired by many contributions in the Computer Graphics and Vision community. I acknowledge them for their work and transparent academic writing."
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
    "text": "If you find ImageQuilting.jl useful in your work, please consider citing the following paper:@ARTICLE{Hoffimann2017,\n  title={Stochastic Simulation by Image Quilting of Deterministic Process-based Geological Models},\n  author={J{\\'u}lio Hoffimann and C{\\'e}line Scheidt and Adrian Barfod and Jef Caers},\n  journal={Computers \\& Geosciences},\n  year={2017}\n}"
},

]}
