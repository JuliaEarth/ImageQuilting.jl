# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function boykov_kolmogorov_cut(A::AbstractArray, B::AbstractArray, dir::Symbol)
  # permute dimensions so that the algorithm is
  # the same for cuts in x, y and z directions
  if dir == :x
    A = permutedims(A, [1,2,3])
    B = permutedims(B, [1,2,3])
  elseif dir == :y
    A = permutedims(A, [2,1,3])
    B = permutedims(B, [2,1,3])
  elseif dir == :z
    A = permutedims(A, [3,2,1])
    B = permutedims(B, [3,2,1])
  end

  E = abs.(A - B)

  mx, my, mz = size(E)
  nvox = mx*my*mz

  # compute gradients
  ∇xₐ = similar(E); ∇yₐ = similar(E); ∇zₐ = similar(E)
  ∇xᵦ = similar(E); ∇yᵦ = similar(E); ∇zᵦ = similar(E)
  for k=1:mz, j=1:my, i=1:mx-1
    ∇xₐ[i,j,k] = A[i+1,j,k] - A[i,j,k]
    ∇xᵦ[i,j,k] = B[i+1,j,k] - B[i,j,k]
  end
  for k=1:mz, j=1:my-1, i=1:mx
    ∇yₐ[i,j,k] = A[i,j+1,k] - A[i,j,k]
    ∇yᵦ[i,j,k] = B[i,j+1,k] - B[i,j,k]
  end
  for k=1:mz-1, j=1:my, i=1:mx
    ∇zₐ[i,j,k] = A[i,j,k+1] - A[i,j,k]
    ∇zᵦ[i,j,k] = B[i,j,k+1] - B[i,j,k]
  end
  if mx > 1
    ∇xₐ[mx,:,:] = ∇xₐ[mx-1,:,:]
    ∇xᵦ[mx,:,:] = ∇xᵦ[mx-1,:,:]
  end
  if my > 1
    ∇yₐ[:,my,:] = ∇yₐ[:,my-1,:]
    ∇yᵦ[:,my,:] = ∇yᵦ[:,my-1,:]
  end
  if mz > 1
    ∇zₐ[:,:,mz] = ∇zₐ[:,:,mz-1]
    ∇zᵦ[:,:,mz] = ∇zᵦ[:,:,mz-1]
  end
  map!(abs, ∇xₐ, ∇xₐ); map!(abs, ∇yₐ, ∇yₐ); map!(abs, ∇zₐ, ∇zₐ)
  map!(abs, ∇xᵦ, ∇xᵦ); map!(abs, ∇yᵦ, ∇yᵦ); map!(abs, ∇zᵦ, ∇zᵦ)

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
    C[c,d] = C[d,c] = (E[c] + E[d]) / (∇xₐ[c] + ∇xₐ[d] + ∇xᵦ[c] + ∇xᵦ[d])
  end
  for k=1:mz, j=1:my-1, i=1:mx
    c = sub2ind((mx,my,mz), i, j, k)
    r = sub2ind((mx,my,mz), i, j+1, k)
    add_edge!(G, c, r)
    add_edge!(G, r, c)
    C[c,r] = C[r,c] = (E[c] + E[r]) / (∇yₐ[c] + ∇yₐ[r] + ∇yᵦ[c] + ∇yᵦ[r])
  end
  for k=1:mz-1, j=1:my, i=1:mx
    c = sub2ind((mx,my,mz), i, j, k)
    o = sub2ind((mx,my,mz), i, j, k+1)
    add_edge!(G, c, o)
    add_edge!(G, o, c)
    C[c,o] = C[o,c] = (E[c] + E[o]) / (∇zₐ[c] + ∇zₐ[o] + ∇zᵦ[c] + ∇zᵦ[o])
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
