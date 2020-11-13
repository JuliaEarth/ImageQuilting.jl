# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function boykov_kolmogorov_cut(A::AbstractArray{T,N}, B::AbstractArray{T,N}, dim::Integer) where {N,T}
  @assert size(A) == size(B) "arrays must have the same size for cut"

  # size and number of voxels
  sz   = size(A)
  nvox = prod(sz)

  # source and sink terminals
  s = nvox + 1; t = nvox + 2

  # lattice graph and capacity matrix
  G = DiGraph(nvox+2)
  C = spzeros(nvox+2, nvox+2)

  # fill lattice graph with original vertices
  for d=1:N
    # loop over all indices except the borders using an increment direction
    inds = CartesianIndices(ntuple(i -> i == d ? (1:sz[d]-1) : (1:sz[i]), N))
    incr = CartesianIndex(ntuple(i -> i == d ? 1 : 0, N))

    for ind in inds
      # adjacent vertices along direction
      u = cart2lin(sz, ind)
      v = cart2lin(sz, ind + incr)

      # simple difference
      Du = abs(A[u] - B[u])
      Dv = abs(A[v] - B[v])

      # gradient along direction
      ∇Au = abs(A[v] - A[u])
      ∇Bu = abs(B[v] - B[u])

      # next vertex along direction
      w = ind + 2incr

      # repeat gradient on the border when
      # outside valid indices for arrays
      ok = all(w.I .≤ sz)
      ∇Av = ok ? abs(A[w] - A[v]) : ∇Au
      ∇Bv = ok ? abs(B[w] - B[v]) : ∇Bu

      add_edge!(G, u, v)
      add_edge!(G, v, u)

      C[u,v] = C[v,u] = (Du + Dv) / (∇Au + ∇Av + ∇Bu + ∇Bv)
    end
  end

  # fill edges from source terminal to left slice
  linds = CartesianIndices(ntuple(i -> i == dim ? 1 : (1:sz[i]), N))
  for ind in linds
    u = cart2lin(sz, ind)
    add_edge!(G, s, u)
    C[s,u] = Inf
  end

  # fill edges from right slice to sink terminal
  rinds = CartesianIndices(ntuple(i -> i == dim ? sz[dim] : (1:sz[i]), N))
  for ind in rinds
    v = cart2lin(sz, ind)
    add_edge!(G, v, t)
    C[v,t] = Inf
  end

  # Boykov-Kolmogorov max-flow/min-cut
  _, __, labels = maximum_flow(G, s, t, C, algorithm=BoykovKolmogorovAlgorithm())

  # remove source and sink terminals
  labels = labels[1:end-2]

  # cut mask
  M = falses(size(A))
  M[labels .== 1] .= true
  M[labels .== 0] .= true

  M
end
