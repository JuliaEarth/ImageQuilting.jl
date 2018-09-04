# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    voxelreuse(trainimg::AbstractArray{T,N}, tilesize::NTuple{N,Int};
               overlapx::Real=1/6, overlapy::Real=1/6, overlapz::Real=1/6,
               nreal::Integer=10, kwargs...)

Returns the mean voxel reuse in `[0,1]` and its standard deviation.

### Notes

- The approximation gets better as `nreal` is made larger.
- Keyword arguments `kwargs` are passed to `iqsim` directly.
"""
function voxelreuse(trainimg::AbstractArray{T,N}, tilesize::NTuple{N,Int};
                    overlapx::Real=1/6, overlapy::Real=1/6, overlapz::Real=1/6,
                    nreal::Integer=10, kwargs...) where {T,N}

  # calculate the overlap from given percentage
  ovx = ceil(Int, overlapx * tilesize[1])
  ovy = ceil(Int, overlapy * tilesize[2])
  ovz = ceil(Int, overlapz * tilesize[3])

  # elementary raster path
  ntilex = ovx > 1 ? 2 : 1
  ntiley = ovy > 1 ? 2 : 1
  ntilez = ovz > 1 ? 2 : 1

  # simulation grid dimensions
  gridsizex = ntilex * (tilesize[1] - ovx) + ovx
  gridsizey = ntiley * (tilesize[2] - ovy) + ovy
  gridsizez = ntilez * (tilesize[3] - ovz) + ovz

  _, _, voxs = iqsim(trainimg, tilesize, gridsizex, gridsizey, gridsizez;
                     overlapx=overlapx, overlapy=overlapy, overlapz=overlapz,
                     nreal=nreal, debug=true, kwargs...)

  μ = mean(voxs)
  σ = std(voxs, mean=μ)

  μ, σ
end
