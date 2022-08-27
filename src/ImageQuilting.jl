# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuilting

using Meshes
using GeoStatsBase

using Tables
using Graphs
using GraphsFlows
using ImageFiltering
using ImageMorphology
using StatsBase: sample, weights
using ProgressMeter: Progress, next!
using FFTW: set_num_threads
using CpuId: cpucores
using RecipesBase
using Primes
using CUDA
using OpenCL
using CLFFT
const clfft = CLFFT

using Base: @nexprs, @nloops, @nref
using SparseArrays: spzeros
using Statistics: mean, std
using Random
 
using PlatformAware

import GeoStatsBase: preprocess, solvesingle

include("utils.jl")
include("utils_gpu.jl")
include("plotrecipes.jl")
include("relaxation.jl")
include("taumodel.jl")
include("graphcut.jl")
include("iqsim.jl")
include("voxelreuse.jl")
include("geostats.jl")

#include("imfilter.jl")

function __init__()
  include(pkgdir(@__MODULE__) * "/src/imfilter.jl")
end

export
  # functions
  iqsim,
  voxelreuse,

  # geostats solver
  IQ

end
