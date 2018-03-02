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

"""
    voxelreuse(training_image::AbstractArray,
               tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;
               overlapx::Real=1/6, overlapy::Real=1/6, overlapz::Real=1/6,
               cut::Symbol=:boykov, simplex::Bool=false, nreal::Integer=10,
               threads::Integer=CPU_PHYSICAL_CORES, gpu::Bool=false,
			   soft::AbstractVector=[], hard::HardData=HardData(), tol::Real=.1)

Returns the mean voxel reuse in `[0,1]` and its standard deviation.

### Notes

The approximation gets better as `nreal` is made larger.
"""
function voxelreuse(training_image::AbstractArray,
                    tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;
                    overlapx::Real=1/6, overlapy::Real=1/6, overlapz::Real=1/6,
                    cut::Symbol=:boykov, simplex::Bool=false, nreal::Integer=10,
                    threads::Integer=CPU_PHYSICAL_CORES, gpu::Bool=false,
					soft::AbstractVector=[], hard::HardData=HardData(), tol::Real=.1)

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
                     cut=cut, simplex=simplex, nreal=nreal,
                     threads=threads, gpu=gpu, debug=true,
					 soft=soft,hard=hard,tol=tol)

  μ = mean(voxs)
  σ = std(voxs, mean=μ)

  μ, σ
end
