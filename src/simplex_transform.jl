# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function simplex_transform(img::AbstractArray, nvertices::Integer)
  # binary images are trivial
  nvertices == 1 && return [img]

  ncoords = nvertices - 1

  # simplex construction
  vertices = [eye(ncoords) ones(ncoords)*(1-sqrt(ncoords+1))/2]
  center = sum(vertices, 2) / nvertices
  vertices = vertices .- center

  # map 0 to (0,0,...,0)
  vertices = [zeros(ncoords) vertices]
  idx = map(Int, img + 1)

  result = Array{Array}(ncoords)
  for i=1:ncoords
    coords = similar(img)
    coords[:] = vertices[i,idx[:]]
    result[i] = coords
  end

  result
end
