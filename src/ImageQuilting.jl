# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuilting

using Graphs
using GraphsFlows
using ImageFiltering
using ImageMorphology
using StatsBase: sample, weights
using ProgressMeter: Progress, next!
using FFTW: set_num_threads
using CpuId: cpucores
using CUDA

using Base: @nexprs, @nloops, @nref
using SparseArrays: spzeros
using Statistics: mean, std
using Random

include("utils.jl")
include("imfilter.jl")
include("relaxation.jl")
include("taumodel.jl")
include("graphcut.jl")
include("iqsim.jl")
include("voxelreuse.jl")

function __init__()
  # register error hint for visualization functions
  # since this is a recurring issue for new users
  Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
    if exc.f == voxelreuse || exc.f == voxelreuseplot!
      if isnothing(Base.get_extension(ImageQuilting, :ImageQuiltingMakieExt))
        print(
          io,
          """

          Did you import a Makie.jl backend (e.g., GLMakie.jl, CairoMakie.jl) for visualization?

          """
        )
        printstyled(
          io,
          """
          julia> using ImageQuilting
          julia> import GLMakie # or CairoMakie, WGLMakie, etc.
          """,
          color=:cyan,
          bold=true
        )
      end
    end
  end
end

export
  # simulation
  iqsim,

  # voxel reuse
  voxelreuse,
  voxelreuseplot,
  voxelreuseplot!

end
