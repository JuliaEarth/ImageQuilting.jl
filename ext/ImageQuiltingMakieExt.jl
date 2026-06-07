# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module ImageQuiltingMakieExt

using ImageQuilting
using Random

import Makie
import ImageQuilting: voxelreuseplot, voxelreuseplot!

Makie.@recipe VoxelReusePlot (trainimg,) begin
  tmin = nothing
  tmax = nothing
  overlap = (1 / 6, 1 / 6, 1 / 6)
  nreal = 10
  rng = Random.default_rng()
  color = :slategray3
end

Makie.preferred_axis_type(::VoxelReusePlot) = Makie.Axis

Makie.preferred_axis_attributes(_, plot::VoxelReusePlot) = (xlabel="Template size", ylabel="Mean voxel reuse")

function Makie.plot!(plot::VoxelReusePlot)
  # retrieve inputs
  trainimg = plot.trainimg[]
  overlap = plot.overlap[]
  nreal = plot.nreal[]
  rng = plot.rng[]
  color = plot.color[]

  ndims(trainimg) == 3 || throw(ArgumentError("image is not 3D (add ghost dimension in 2D)"))

  # choose tile size range
  tmin, tmax, idx = tminmax(plot)

  # save support as an array
  ts = collect(tmin:tmax)

  # compute voxel reuse for each tile size
  μσ = map(ts) do t
    tilesize = ntuple(i -> idx[i] ? t : 1, 3)
    voxelreuse(trainimg, tilesize; overlap=overlap, nreal=nreal, rng=rng, showprogress=false)
  end
  μs = first.(μσ)
  σs = last.(μσ)

  # optimal tile size range
  rank = sortperm(μs, rev=true)
  best = ts[rank[1:min(5, length(ts))]]
  t₋, t₊ = minimum(best), maximum(best)

  Makie.vlines!(plot, [t₋, t₊], linestyle=:dash, color=color)
  Makie.band!(plot, ts, μs - σs, μs + σs, alpha=0.5, color=color)
  Makie.lines!(plot, ts, μs, color=color)
end

function Makie.data_limits(plot::VoxelReusePlot)
  tmin, tmax, idx = tminmax(plot)
  pmin = Makie.Point3f(0, 0, 0)
  pmax = Makie.Point3f(tmax, 1, 0)
  Makie.Rect3f([pmin, pmax])
end

function tminmax(plot::VoxelReusePlot)
  # retrieve inputs
  trainimg = plot.trainimg[]
  tmin = plot.tmin[]
  tmax = plot.tmax[]

  # image size
  dims = size(trainimg)
  idx = [dims...] .> 1

  # default support
  isnothing(tmin) && (tmin = 7)
  isnothing(tmax) && (tmax = min(100, minimum(dims[idx])))

  tmin > 0 || throw(ArgumentError("`tmin` must be positive"))
  tmin < tmax || throw(ArgumentError("`tmin` must be smaller than `tmax`"))

  tmin, tmax, idx
end

end
