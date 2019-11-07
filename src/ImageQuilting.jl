# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuilting

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

include("utils.jl")
include("plot_recipes.jl")
include("imfilter_cpu.jl")
include("relaxation.jl")
include("tau_model.jl")
include("graphcuts.jl")
include("voxel_reuse.jl")
include("iqsim.jl")

# optionally load GeoStats.jl API
using Requires
function __init__()
    @require GeoStatsBase="323cb8eb-fbf6-51c0-afd0-f8fba70507b2" include("geostats.jl")
end

export
  # functions
  iqsim,
  voxelreuse,

  # geostats solver
  ImgQuilt

end
