# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuilting

using ImageFiltering
using ImageMorphology
using LightGraphs
using LightGraphsFlows
using Base: @nexprs, @nloops, @nref
using Combinatorics: nthperm!
using StatsBase: sample, weights
using Primes: factor
using ProgressMeter: Progress, next!
using CpuId: cpucores
using LinearAlgebra: I # TODO: eliminate this dependency
using SparseArrays: spzeros
using Random: shuffle!
using Statistics: mean, std
using RecipesBase

# GeoStats.jl interface
using GeoStatsBase
using GeoStatsDevTools

import GeoStatsBase: preprocess, solve_single

include("utils.jl")
# include("utils_gpu.jl")
include("datatypes.jl")
include("plot_recipes.jl")
include("imfilter_cpu.jl")
# include("imfilter_gpu.jl")
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
  ImgQuilt

end
