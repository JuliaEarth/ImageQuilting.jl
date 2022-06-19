# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuilting

using Meshes
using GeoStatsBase

using Graphs
using GraphsFlows
using ImageFiltering
using ImageMorphology
using Primes: factor
using StatsBase: sample, weights
using ProgressMeter: Progress, next!
using FFTW: set_num_threads
using CpuId: cpucores
using RecipesBase
using Base: @nexprs, @nloops, @nref
using SparseArrays: spzeros
using Statistics: mean, std
using Random

import GeoStatsBase: preprocess, solvesingle

include("utils.jl")
include("plotrecipes.jl")
include("imfilter.jl")
include("relaxation.jl")
include("taumodel.jl")
include("graphcuts.jl")
include("voxelreuse.jl")
include("iqsim.jl")
include("geostats.jl")

export
  # functions
  iqsim,
  voxelreuse,

  # geostats solver
  IQ

end
