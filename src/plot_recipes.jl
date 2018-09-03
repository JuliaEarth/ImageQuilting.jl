# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

@userplot VoxelReusePlot

@recipe function f(vr::VoxelReusePlot; tmin=nothing, tmax=nothing,
                   overlapx=1/6, overlapy=1/6, overlapz=1/6,
                   nreal=10, gpu=false)
  # get input image
  img = vr.args[1]

  @assert ndims(img) == 3 "image is not 3D (add ghost dimension for 2D)"

  # image extent
  extent = size(img)
  idx = [extent...] .> 1

  # default support
  tmin == nothing && (tmin = 7)
  tmax == nothing && (tmax = min(100, minimum(extent[idx])))

  @assert tmin > 0 "tmin must be positive"
  @assert tmin < tmax "tmin must be smaller than tmax"

  # save support as an array
  ts = collect(tmin:tmax)

  # compute voxel reuse for each template size
  p = Progress(length(ts), color=:black)
  μs = Float64[]; σs = Float64[]
  for t in ts
    tplconfig = [1,1,1]
    tplconfig[idx] .= t

    μ, σ = voxelreuse(img, tplconfig...;
                      overlapx=overlapx,
                      overlapy=overlapy,
                      overlapz=overlapz,
                      nreal=nreal, gpu=gpu)

    push!(μs, μ)
    push!(σs, σ)
    next!(p)
  end

  # highlight the optimum template range
  rank = sortperm(μs, rev=true)
  best = ts[rank[1:min(5,length(ts))]]
  xmin, xmax = minimum(best), maximum(best)

  yref = minimum(μs - σs)
  ymin = max(0., yref - .05)
  ymax = ymin + .008

  @series begin
    seriestype := :shape
    linewidth := 0
    fillalpha := .5
    label --> "Optimum range: [$xmin, $xmax]"

    [xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]
  end

  seriestype := :path
  primary := false
  ribbon := σs
  fillalpha := .5
  xlim := (0, tmax)
  ylim := (ymin, Inf)
  xlabel --> "Template size"
  ylabel --> "Voxel reuse"

  ts, μs
end
