## Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
##
## Permission to use, copy, modify, and/or distribute this software for any
## purpose with or without fee is hereby granted, provided that the above
## copyright notice and this permission notice appear in all copies.
##
## THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
## WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
## ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
## WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
## ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
## OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# Julia doesn't support optional dependencies yet. We have
# to disable precompilation in order to avoid issues with
# missing components at runtime.
__precompile__(false)

module ImageQuilting

using Images
using LightGraphs
using RecipesBase
using Base: @nexprs, @nloops, @nref
using Combinatorics: nthperm!
using StatsBase: sample, weights
using Primes: factor
using ProgressMeter: Progress, next!
using Hwloc: topology_load, histmap

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

export
  # functions
  iqsim,
  voxelreuse,

  # types
  SoftData,
  HardData,
  VoxelReuse

end
