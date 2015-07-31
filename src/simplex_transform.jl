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

function simplex_transform(img::AbstractArray, nvertices::Integer)
  # binary images are trivial
  nvertices == 1 && return Any[img]

  ncoords = nvertices - 1

  # simplex construction
  vertices = [eye(ncoords) ones(ncoords)*(1-sqrt(ncoords+1))/2]
  center = sum(vertices, 2) / nvertices
  vertices -= repeat(center, inner=[1,nvertices])

  # map 0 to (0,0,...,0)
  vertices = [zeros(ncoords) vertices]
  idx = map(Int, img + 1)

  result = cell(ncoords)
  for i=1:ncoords
    coords = similar(img)
    coords[:] = vertices[i,idx[:]]
    result[i] = coords
  end

  result
end
