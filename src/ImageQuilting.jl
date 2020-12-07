# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuilting

using GeoStatsBase
using ImageFiltering
using ImageMorphology
using LightGraphs
using LightGraphsFlows
using Base: @nexprs, @nloops, @nref
using Primes: factor
using StatsBase: sample, weights
using ProgressMeter: Progress, next!
using FFTW: set_num_threads
using CpuId: cpucores
using RecipesBase
using SparseArrays: spzeros
using Random: shuffle!, randperm
using Statistics: mean, std

import GeoStatsBase: preprocess, solvesingle

include("utils.jl")
include("plot_recipes.jl")
include("imfilter_cpu.jl")
include("relaxation.jl")
include("tau_model.jl")
include("graphcuts.jl")
include("voxel_reuse.jl")
include("iqsim.jl")
include("geostats.jl")

export
  # functions
  iqsim,
  voxelreuse,

  # geostats solver
  IQ

end
