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

function boundary_cut(A::AbstractArray, B::AbstractArray, dir::Symbol)
  # permute overlap cube dimensions so that the algorithm
  # is the same for cuts in x, y and z directions.
  E = abs(A - B)
  if dir == :x
    E = permutedims(E, [1,2,3])
  elseif dir == :y
    E = permutedims(E, [2,1,3])
  elseif dir == :z
    E = permutedims(E, [3,2,1])
  end

  mx, my, mz = size(E)
  nvox = mx*my*mz

  # add source and sink terminals
  s = nvox + 1; t = nvox + 2

  # construct graph and capacity matrix
  G = DiGraph(nvox+2)
  C = spzeros(nvox+2, nvox+2)
  for k=1:mz, j=1:my, i=1:mx-1
    c = sub2ind((mx,my,mz), i, j, k)
    d = sub2ind((mx,my,mz), i+1, j, k)
    add_edge!(G, c, d)
    add_edge!(G, d, c)
    C[c,d] = C[d,c] = E[c] + E[d]
  end
  for k=1:mz, j=1:my-1, i=1:mx
    c = sub2ind((mx,my,mz), i, j, k)
    r = sub2ind((mx,my,mz), i, j+1, k)
    add_edge!(G, c, r)
    add_edge!(G, r, c)
    C[c,r] = C[r,c] = E[c] + E[r]
  end
  for k=1:mz-1, j=1:my, i=1:mx
    c = sub2ind((mx,my,mz), i, j, k)
    o = sub2ind((mx,my,mz), i, j, k+1)
    add_edge!(G, c, o)
    add_edge!(G, o, c)
    C[c,o] = C[o,c] = E[c] + E[o]
  end
  for k=1:mz, j=1:my
    u = sub2ind((mx,my,mz), 1, j, k)
    v = sub2ind((mx,my,mz), mx, j, k)
    add_edge!(G, s, u)
    add_edge!(G, v, t)
    C[s,u] = C[v,t] = Inf
  end

  # Boykov-Kolmogorov max-flow/min-cut
  _, __, labels = maximum_flow(G, s, t, C, algorithm=BoykovKolmogorovAlgorithm())

  # remove source and sink terminals
  labels = labels[1:end-2]

  # cut mask
  M = falses(E)
  M[labels .== 1] = true
  M[labels .== 0] = true

  # permute back to original shape
  if dir == :x
    M = permutedims(M, [1,2,3])
  elseif dir == :y
    M = permutedims(M, [2,1,3])
  elseif dir == :z
    M = permutedims(M, [3,2,1])
  end

  M
end
