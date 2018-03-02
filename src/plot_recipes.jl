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

@userplot VoxelReusePlot

@recipe function f(vr::VoxelReusePlot; 
        tmin=nothing, tmax=nothing, nreal=10, 
        gpu=false, soft=[], hard=HardData(), tol=.1,
        overlapx=1/6, overlapy=1/6, overlapz=1/6, 
        cut=:boykov, simplex=false, threads=CPU_PHYSICAL_CORES)
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

  # -- compute voxel reuse
  extent = size(img)
  idx = [extent...] .> 1

  p = Progress(length(ts), color=:black)
  μs = Float64[]; σs = Float64[]
  for T in ts
    tplconfig = [1,1,1]
    tplconfig[idx] = T
    μ, σ = voxelreuse(img, tplconfig...,
            nreal=nreal, gpu=gpu, soft=soft, hard=hard,
			overlapx=overlapx, overlapy=overlapy, overlapz=overlapz,
			cut=cut, simplex=simplex, threads=threads)
    push!(μs, μ); push!(σs, σ)

    next!(p)
  end
  # --

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

