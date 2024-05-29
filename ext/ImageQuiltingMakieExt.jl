# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuiltingMakieExt

using ImageQuilting
using Random

import Makie
import ImageQuilting: voxelreuseplot, voxelreuseplot!

Makie.@recipe(VoxelReusePlot, trainimg) do scene
  Makie.Attributes(tmin=nothing, tmax=nothing, overlap=(1 / 6, 1 / 6, 1 / 6), nreal=10, rng=Random.default_rng())
end

Makie.preferred_axis_type(::VoxelReusePlot) = Makie.Axis

function Makie.plot!(plot::VoxelReusePlot)
  # retrieve inputs
  trainimg = plot[:trainimg][]
  overlap = plot[:overlap][]
  nreal = plot[:nreal][]
  rng = plot[:rng][]

  ndims(trainimg) == 3 || throw(ArgumentError("image is not 3D (add ghost dimension in 2D)"))

  # choose tile size range
  tmin, tmax, idx = tminmax(plot)

  # save support as an array
  ts = collect(tmin:tmax)

  # compute voxel reuse for each tile size
  μσ = map(ts) do t
    tilesize = ntuple(i -> idx[i] ? t : 1, 3)
    voxelreuse(trainimg, tilesize; overlap=overlap, nreal=nreal, rng=rng)
  end
  μs = first.(μσ)
  σs = last.(μσ)

  # optimal tile size range
  rank = sortperm(μs, rev=true)
  best = ts[rank[1:min(5, length(ts))]]
  t₋, t₊ = minimum(best), maximum(best)

  Makie.vspan!(plot, [t₋], [t₊], alpha=0.5, color=:slategray3)
  Makie.band!(plot, ts, μs - σs, μs + σs, alpha=0.5, color=:slategray3)
  Makie.lines!(plot, ts, μs, color=:slategray3)
end

function Makie.data_limits(plot::VoxelReusePlot)
  tmin, tmax, idx = tminmax(plot)
  pmin = Makie.Point3f(0, 0, 0)
  pmax = Makie.Point3f(tmax, 1, 0)
  Makie.Rect3f([pmin, pmax])
end

function tminmax(plot::VoxelReusePlot)
  # retrieve inputs
  trainimg = plot[:trainimg][]
  tmin = plot[:tmin][]
  tmax = plot[:tmax][]

  # image size
  dims = size(trainimg)
  idx = [dims...] .> 1

  # default support
  isnothing(tmin) && (tmin = 7)
  isnothing(tmax) && (tmax = min(100, minimum(dims[idx])))

  tmin > 0 || throw(ArgumentError("`tmin` must be positive"))
  tmin < tmax || throw(ArgumentError("`tmin`` must be smaller than `tmax`"))

  tmin, tmax, idx
end

end
