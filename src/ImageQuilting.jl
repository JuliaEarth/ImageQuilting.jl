# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

# Julia doesn't support optional dependencies yet. We have
# to disable precompilation in order to avoid issues with
# missing components at runtime.
__precompile__(false)

module ImageQuilting

using ImageFiltering
using ImageMorphology
using LightGraphs
using LightGraphsFlows
using RecipesBase
using Base: @nexprs, @nloops, @nref
using Combinatorics: nthperm!
using StatsBase: sample, weights
using Primes: factor
using ProgressMeter: Progress, next!
using Hwloc: num_physical_cores

# optional dependencies
try
  using OpenCL
catch
  global cl = nothing
end

try
  using CLFFT
  global clfft = CLFFT
catch
  global clfft = nothing
end

# GeoStats.jl interface
importall GeoStatsBase
using GeoStatsDevTools

include("utils.jl")
include("utils_gpu.jl")
include("datatypes.jl")
include("plot_recipes.jl")
include("imfilter_cpu.jl")
include("imfilter_gpu.jl")
include("relaxation.jl")
include("tau_model.jl")
include("dijkstra_cut.jl")
include("boykov_kolmogorov_cut.jl")
include("simplex_transform.jl")
include("voxel_reuse.jl")
include("iqsim.jl")

include("geostats_api.jl")

export
  # functions
  iqsim,
  voxelreuse,

  # data types
  HardData,

  # geostats solver
  ImgQuilt,

  # deprecated
  SoftData

end
