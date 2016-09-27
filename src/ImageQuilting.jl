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

__precompile__()

module ImageQuilting

using Base: @nexprs, @nloops, @nref
using Images: imfilter_fft, padarray, dilate
using StatsBase: sample, weights
using LightGraphs

if VERSION > v"0.5-"
  using Combinatorics: nthperm!
else
  global view = slice
end

try # optional dependencies
  using OpenCL
  using CLFFT
  global cl = OpenCL
  global clfft = CLFFT
catch
  global cl = nothing
  global clfft = nothing
end

include("utils.jl")
include("utils_gpu.jl")
include("datatypes.jl")
include("imfilter_gpu.jl")
include("relaxation.jl")
include("tau_model.jl")
include("dijkstra_cut.jl")
include("boykov_kolmogorov_cut.jl")
include("simplex_transform.jl")
include("mean_voxel_reuse.jl")
include("iqsim.jl")

export
  # functions
  iqsim,
  meanvoxreuse,

  # types
  SoftData,
  HardData

end
