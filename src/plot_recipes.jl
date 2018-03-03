# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

@userplot VoxelReusePlot

@recipe function f(vr::VoxelReusePlot; tmin=nothing, tmax=nothing, nreal=10)
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
    tplconfig[idx] = t

    μ, σ = voxelreuse(img, tplconfig...;
                      nreal=nreal)

    push!(μs, μ)
    push!(σs, σ)
    next!(p)
  end

  # highlight the optimum template range
  @series begin
    rank = sortperm(μs, rev=true)
    best = ts[rank[1:min(5,length(ts))]]
    xmin, xmax = minimum(best), maximum(best)

    reference = minimum(μs - σs)
    ymin = max(0., reference - .05)
    ymax = ymin + .008

    tplrange = [xmin, xmax]

    seriestype := :shape
    linewidth := 0
    fillalpha := .5
    label --> "Optimum range: $tplrange"

    [xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax]
  end

  seriestype := :path
  primary := false
  ribbon := σs
  fillalpha := .5
  xlim := (0, tmax)
  xlabel --> "Template size"
  ylabel --> "Voxel reuse"

  ts, μs
end
