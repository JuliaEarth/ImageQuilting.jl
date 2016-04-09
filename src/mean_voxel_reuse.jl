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

function meanvoxreuse(training_image::AbstractArray,
                      tplsizex::Integer, tplsizey::Integer, tplsizez::Integer;
                      overlapx=1/6, overlapy=1/6, overlapz=1/6,
                      nreal=10, cut=:dijkstra, categorical=false)

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

  _, _, voxs = iqsim(training_image, tplsizex, tplsizey, tplsizez,
                     gridsizex, gridsizey, gridsizez,
                     overlapx=overlapx, overlapy=overlapy, overlapz=overlapz,
                     nreal=nreal, cut=cut, categorical=categorical, debug=true)

  mean(voxs)
end
