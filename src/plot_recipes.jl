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

struct VoxelReuse
  img::AbstractArray

  VoxelReuse(img) = ndims(img) == 3 ? new(img) : error("Image must be a 3D array")
end

@recipe function f(vr::VoxelReuse; tmin=nothing, tmax=nothing, nreal=10)
  extent = size(vr.img)
  idx = [extent...] .> 1

  # default support
  tmin == nothing && (tmin = 7)
  tmax == nothing && (tmax = min(100, minimum(extent[idx])))

  @assert tmin > 0 "tmin must be positive"
  @assert tmin < tmax "tmin must be smaller than tmax"

  # save support as an array
  ts = collect(tmin:tmax)

  # compute voxel reuse
  μs, σs = mapreuse(vr.img, ts, nreal)

  # highlight the optimum template range
  @series begin
    rank = sortperm(μs, rev=true)
    best = ts[rank[1:min(5,length(ts))]]
    xmin, xmax = minimum(best), maximum(best)

    reference = μs[1] - σs[1]
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

function mapreuse(img::AbstractArray, ts::Array{Int}, nreal::Int)
  extent = size(img)
  idx = [extent...] .> 1

  p = Progress(length(ts), color=:black)
  μs = Float64[]; σs = Float64[]
  for T in ts
    tplconfig = [1,1,1]
    tplconfig[idx] = T
    μ, σ = voxelreuse(img, tplconfig..., nreal=nreal)
    push!(μs, μ); push!(σs, σ)

    next!(p)
  end

  μs, σs
end
