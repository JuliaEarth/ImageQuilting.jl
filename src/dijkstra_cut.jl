# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function dijkstra_cut(A1::AbstractArray, A2::AbstractArray, dir::Symbol)
  # permute dimensions so that the algorithm is
  # the same for cuts in x, y and z directions
  B = abs.(A1 - A2)
  if dir == :x
    B = permutedims(B, [1,2,3])
  elseif dir == :y
    B = permutedims(B, [2,1,3])
  elseif dir == :z
    B = permutedims(B, [3,2,1])
  end

  mx, my, mz = size(B)

  # accumulation cube and overlap mask
  E = zeros(B); M = falses(B)

  # pad accumulation cube with +inf
  Epad(i,j,k) = all(0 .< [i,j,k] .≤ [mx,my,mz]) ? E[i,j,k] : Inf

  # forward accumulation along 3D cube
  E[:,:,1] = B[:,:,1]
  for k=2:mz, i=1:mx, j=1:my
    square = [Epad(i-1 , j-1 , k-1) , Epad(i-1 , j , k-1) , Epad(i-1 , j+1 , k-1),
              Epad(i   , j-1 , k-1) , Epad(i   , j , k-1) , Epad(i   , j+1 , k-1),
              Epad(i+1 , j-1 , k-1) , Epad(i+1 , j , k-1) , Epad(i+1 , j+1 , k-1)]
    E[i,j,k] = B[i,j,k] + minimum(square)
  end

  # forward accumulation along 2D square (last slice of the cube)
  zslice = view(E,:,:,mz)
  for j=2:my
    zslice[1,j] += minimum(zslice[1:2,j-1])
    for i=2:mx-1
      zslice[i,j] += minimum(zslice[i-1:i+1,j-1])
    end
    zslice[mx,j] += minimum(zslice[mx-1:mx,j-1])
  end

  # backward search along last slice
  idx = indmin(zslice[:,my])
  mslice = view(M,:,:,mz)
  mslice[1:idx,my] = trues(idx)
  idxvec = zeros(Int, my); idxvec[my] = idx # keep track of indexes
  width = isodd(mx) ? mx+1 : mx # avoid zig-zag artifact
  for j=my-1:-1:1
    for i=1:width
      if idx < mx && minimum(zslice[max(idx-1,1):idx+1,j]) == zslice[idx+1,j]
        idx += 1
      elseif idx > 1 && zslice[idx-1,j] ≤ zslice[idx,j]
        idx -= 1
      end
    end
    mslice[1:idx,j] = true
    idxvec[j] = idx
  end

  # backward search along cube
  for j=1:my
    yslice = view(E,:,j,:)
    idx = idxvec[j]
    for k=mz-1:-1:1
      for i=1:width
        if idx < mx && minimum(yslice[max(idx-1,1):idx+1,k]) == yslice[idx+1,k]
          idx += 1
        elseif idx > 1 && yslice[idx-1,k] ≤ yslice[idx,k]
          idx -= 1
        end
      end
      M[1:idx,j,k] = true
    end
  end

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
