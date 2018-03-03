# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    voxelreuse(training_image::AbstractArray,
               tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;
               overlapx::Real=1/6, overlapy::Real=1/6, overlapz::Real=1/6,
               nreal::Integer=10, kwargs...)

Returns the mean voxel reuse in `[0,1]` and its standard deviation.

### Notes

- The approximation gets better as `nreal` is made larger.
- Keyword arguments `kwargs` are passed to `iqsim` directly.
"""
function voxelreuse(training_image::AbstractArray,
                    tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;
                    overlapx::Real=1/6, overlapy::Real=1/6, overlapz::Real=1/6,
                    nreal::Integer=10, kwargs...)

  # calculate the overlap from given percentage
  ovx = ceil(Int, overlapx * tplsizex)
  ovy = ceil(Int, overlapy * tplsizey)
  ovz = ceil(Int, overlapz * tplsizez)

  # elementary raster path
  ntilex = ovx > 1 ? 2 : 1
  ntiley = ovy > 1 ? 2 : 1
  ntilez = ovz > 1 ? 2 : 1

  # simulation grid dimensions
  gridsizex = ntilex * (tplsizex - ovx) + ovx
  gridsizey = ntiley * (tplsizey - ovy) + ovy
  gridsizez = ntilez * (tplsizez - ovz) + ovz

  _, _, voxs = iqsim(training_image,
                     tplsizex, tplsizey, tplsizez,
                     gridsizex, gridsizey, gridsizez,
                     overlapx=overlapx, overlapy=overlapy, overlapz=overlapz,
                     nreal=nreal, debug=true, kwargs...)

  μ = mean(voxs)
  σ = std(voxs, mean=μ)

  μ, σ
end
