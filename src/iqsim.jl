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

using Images: imfilter_fft, padarray, dilate

immutable SoftData
  data::AbstractArray
  transform::Function
end

typealias HardData Dict{NTuple{3,Integer},Real}

function iqsim(training_image::AbstractArray,
               tplsizex::Integer, tplsizey::Integer, tplsizez::Integer,
               gridsizex::Integer, gridsizey::Integer, gridsizez::Integer;
               overlapx=1/6, overlapy=1/6, overlapz=1/6,
               seed=0, nreal=1, cutoff=.1, categorical=false,
               soft=nothing, hard=nothing, debug=false)

  # sanity checks
  @assert ndims(training_image) == 3 "training image is not 3D (add ghost dimension for 2D)"
  @assert all(0 .< [tplsizex, tplsizey, tplsizez] .≤ [size(training_image)...]) "invalid template size"
  @assert all([gridsizex, gridsizey, gridsizez] .≥ [tplsizex, tplsizey, tplsizez]) "invalid grid size"
  @assert all(0 .< [overlapx, overlapy, overlapz] .< 1) "overlaps must be in range (0,1)"
  @assert cutoff > 0 "cutoff must be positive"

  # soft data checks
  if soft ≠ nothing
    @assert isa(soft, SoftData)
    @assert ndims(soft.data) == 3 "soft data is not 3D (add ghost dimension for 2D)"
    @assert all([size(soft.data)...] .≥ [gridsizex, gridsizey, gridsizez]) "soft data size < grid size"
    @assert 0 < cutoff ≤ 1 "cutoff must be in range (0,1] when soft data is available"
  end

  # hard data checks
  if hard ≠ nothing
    @assert isa(hard, HardData)
    locations = Int[loc[i] for loc in keys(hard), i=1:3]
    @assert all(maximum(locations, 1) .≤ [gridsizex gridsizey gridsizez]) "hard data locations outside of grid"
    @assert all(minimum(locations, 1) .> 0) "hard data locations must be positive indexes"
  end

  # calculate the overlap from given percentage
  overlapx = ceil(Int, overlapx * tplsizex)
  overlapy = ceil(Int, overlapy * tplsizey)
  overlapz = ceil(Int, overlapz * tplsizez)

  # spacing in raster path
  spacingx = tplsizex - overlapx
  spacingy = tplsizey - overlapy
  spacingz = tplsizez - overlapz

  # calculate the number of tiles from grid size
  ntilex = ceil(Int, gridsizex / max(spacingx, 1))
  ntiley = ceil(Int, gridsizey / max(spacingy, 1))
  ntilez = ceil(Int, gridsizez / max(spacingz, 1))

  # always work with floating point
  TI = map(Float64, training_image)

  # perform simplex transform
  simplexTI = Any[TI]; nvertices = 1
  if categorical
    categories = Set(training_image)
    ncategories = nvertices = length(categories) - 1

    @assert categories == Set(0:ncategories) "Categories should be labeled 1, 2, 3,..."

    simplexTI = simplex_transform(TI, nvertices)
  end

  # simulation grid dimensions
  nx = ntilex * (tplsizex - overlapx) + overlapx
  ny = ntiley * (tplsizey - overlapy) + overlapy
  nz = ntilez * (tplsizez - overlapz) + overlapz

  # total overlap volume in simulation grid
  overlap_volume = nx*ny*nz - (nx - (ntilex-1)overlapx)*
                              (ny - (ntiley-1)overlapy)*
                              (nz - (ntilez-1)overlapz)

  # pad soft data and soft transform training image
  softgrid = softTI = []
  if soft ≠ nothing
    mx, my, mz = size(soft.data)
    lx = min(mx,nx); ly = min(my,ny); lz = min(mz,nz)

    softgrid = padarray(soft.data, [0,0,0], [nx-lx,ny-ly,nz-lz], "symmetric")

    softTI = soft.transform(TI)
  end

  # simulation grid irregularities
  activated = []
  if hard ≠ nothing
    activated = trues(nx, ny, nz)
    for loc in keys(hard)
      if isnan(hard[loc])
        activated[loc...] = false
      end
    end
  end

  # main output is a vector of 3D grids
  realizations = []

  # for each realization we have:
  boundarycuts = [] # boundary cut
  voxelreusage = [] # voxel reusage

  srand(seed)

  for real=1:nreal
    # allocate memory for current simulation
    simgrid = zeros(nx, ny, nz)
    cutgrid = debug ? falses(nx, ny, nz) : []

    # set hard data
    simulated = []
    if hard ≠ nothing
      simulated = falses(nx, ny, nz)
      for loc in keys(hard)
        if !isnan(hard[loc])
          simgrid[loc...] = hard[loc]
          simulated[loc...] = true
        end
      end
    end

    # keep track of skipped tiles
    skipped = Set{NTuple{3,Int}}()

    # loop simulation grid tile by tile
    for i=1:ntilex, j=1:ntiley, k=1:ntilez
      # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
      iₛ = (i-1)spacingx + 1
      jₛ = (j-1)spacingy + 1
      kₛ = (k-1)spacingz + 1
      iₑ = iₛ + tplsizex - 1
      jₑ = jₛ + tplsizey - 1
      kₑ = kₛ + tplsizez - 1

      # skip tile if it contains hard data
      if hard ≠ nothing
        for loc in keys(hard)
          if all([iₛ,jₛ,kₛ] .≤ [loc...] .≤ [iₑ,jₑ,kₑ])
            push!(skipped, (i,j,k))
            break
          end
        end
        (i,j,k) ∈ skipped && continue
      end

      # current simulation dataevent
      simdev = simgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]

      # compute the distance between the simulation dataevent
      # and all patterns in the training image
      mₜ, nₜ, pₜ = size(TI)
      distance = zeros(mₜ-tplsizex+1, nₜ-tplsizey+1, pₜ-tplsizez+1)
      if i > 1 && overlapx > 1 && (i-1,j,k) ∉ skipped
        ovx = simdev[1:overlapx,:,:]
        xsimplex = categorical ? simplex_transform(ovx, nvertices) : Any[ovx]

        D = convdist(simplexTI, xsimplex)
        distance += D[1:mₜ-tplsizex+1,:,:]
      end
      if j > 1 && overlapy > 1 && (i,j-1,k) ∉ skipped
        ovy = simdev[:,1:overlapy,:]
        ysimplex = categorical ? simplex_transform(ovy, nvertices) : Any[ovy]

        D = convdist(simplexTI, ysimplex)
        distance += D[:,1:nₜ-tplsizey+1,:]
      end
      if k > 1 && overlapz > 1 && (i,j,k-1) ∉ skipped
        ovz = simdev[:,:,1:overlapz]
        zsimplex = categorical ? simplex_transform(ovz, nvertices) : Any[ovz]

        D = convdist(simplexTI, zsimplex)
        distance += D[:,:,1:pₜ-tplsizez+1]
      end

      # current pattern database
      patterndb = []
      if soft ≠ nothing
        # compute the distance between the soft dataevent and
        # all dataevents in the soft training image
        softdev = softgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        softdistance = convdist(Any[softTI], Any[softdev])

        # candidates with good overlap
        dbsize = ceil(Int, cutoff*length(distance))
        idx1 = sortperm(distance[:])[1:dbsize]

        softcutoff = .1
        while true
          # candidates in accordance with soft data
          softdbsize = ceil(Int, softcutoff*length(softdistance))
          idx2 = sortperm(softdistance[:])[1:softdbsize]

          patterndb = intersect(idx1, idx2)

          !isempty(patterndb) && break
          softcutoff += .1
        end
      else
        patterndb = find(distance .≤ (1+cutoff)minimum(distance))
      end

      # pick a pattern at random from the database
      idx = patterndb[rand(1:length(patterndb))]
      iᵦ, jᵦ, kᵦ = ind2sub(size(distance), idx)

      # selected training image dataevent
      TIdev = TI[iᵦ:iᵦ+tplsizex-1,jᵦ:jᵦ+tplsizey-1,kᵦ:kᵦ+tplsizez-1]

      # minimum boundary cut mask
      M = falses(simdev)
      if i > 1 && overlapx > 1 && (i-1,j,k) ∉ skipped
        Bx = overlapdist(simdev[1:overlapx,:,:], TIdev[1:overlapx,:,:],
                         categorical ? nvertices : -1)
        M[1:overlapx,:,:] |= boundary_cut(Bx, :x)
      end
      if j > 1 && overlapy > 1 && (i,j-1,k) ∉ skipped
        By = overlapdist(simdev[:,1:overlapy,:], TIdev[:,1:overlapy,:],
                         categorical ? nvertices : -1)
        M[:,1:overlapy,:] |= boundary_cut(By, :y)
      end
      if k > 1 && overlapz > 1 && (i,j,k-1) ∉ skipped
        Bz = overlapdist(simdev[:,:,1:overlapz], TIdev[:,:,1:overlapz],
                         categorical ? nvertices : -1)
        M[:,:,1:overlapz] |= boundary_cut(Bz, :z)
      end

      # paste contributions from simulation grid and training image
      simgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M.*simdev + !M.*TIdev

      # simulation progress
      hard ≠ nothing && (simulated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = true)

      # save boundary cut
      debug && (cutgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M)
    end

    # throw away voxels that are outside of the grid
    simgrid = simgrid[1:gridsizex,1:gridsizey,1:gridsizez]

    #-----------------------------------------------------------------

    # simulate remaining voxels skipped during raster path
    if hard ≠ nothing
      tplx, tply, tplz = tplsizex, tplsizey, tplsizez

      # throw away voxels that are outside of the grid
      simulated = simulated[1:gridsizex,1:gridsizey,1:gridsizez]
      activated = activated[1:gridsizex,1:gridsizey,1:gridsizez]

      # morphological dilation
      dilated = dilate(simulated) & activated

      while dilated ≠ simulated
        visited = 0
        for vox in find(dilated - simulated)
          # tile center is given by (iᵥ,jᵥ,kᵥ)
          iᵥ, jᵥ, kᵥ = ind2sub(size(simgrid), vox)

          # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
          iₛ = iᵥ - (tplx-1)÷2
          jₛ = jᵥ - (tply-1)÷2
          kₛ = kᵥ - (tplz-1)÷2
          iₑ = iₛ + tplx - 1
          jₑ = jₛ + tply - 1
          kₑ = kₛ + tplz - 1

          if all(0 .< [iₛ,jₛ,kₛ]) && all([iₑ,jₑ,kₑ] .≤ [size(simgrid)...]) && !simulated[vox]
            # mark location as visited
            visited += 1

            # voxel-centered dataevent
            simdev = simgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
            simplexdev = categorical ? simplex_transform(simdev, nvertices) : Any[simdev]

            # on/off dataevent
            booldev = simulated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]

            # current pattern database
            distance = convdist(simplexTI, simplexdev, weights=booldev, inner=false)
            patterndb = find(distance .≤ (1+cutoff)minimum(distance))

            # pick a pattern at random from the database
            idx = patterndb[rand(1:length(patterndb))]
            iᵦ, jᵦ, kᵦ = ind2sub(size(distance), idx)

            # tile top left corner is given by (Is,Js,Ks)
            Is = iᵦ - (tplx-1)÷2
            Js = jᵦ - (tply-1)÷2
            Ks = kᵦ - (tplz-1)÷2

            # pad training image dataevent
            TIdev = zeros(tplx, tply, tplz)
            valid = falses(tplx, tply, tplz)
            for δi=1:tplx, δj=1:tply, δk=1:tplz
              i, j, k = Is+δi-1, Js+δj-1, Ks+δk-1
              if all(0 .< [i,j,k] .≤ [size(TI)...])
                TIdev[δi,δj,δk] = TI[i,j,k]
                valid[δi,δj,δk] = true
              end
            end

            # voxels to be simulated
            M = valid & activated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] & !simulated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]

            # paste highlighted portion onto simulation grid
            simgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M.*TIdev + !M.*simdev

            # simulation progress
            simulated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] |= M
          end
        end

        if visited == 0
          # reduce the template size and proceed
          tplx, tply, tplz = max(tplx-1, 1), max(tply-1, 1), max(tplz-1, 1)
        end

        dilated = dilate(simulated) & activated
      end

      # arbitrarily shaped simulation grid
      simgrid[!activated] = categorical ? 0 : NaN
    end

    push!(realizations, categorical ? map(Int, simgrid) : simgrid)

    debug && push!(voxelreusage, sum(cutgrid)/overlap_volume)
    debug && (cutgrid = cutgrid[1:gridsizex,1:gridsizey,1:gridsizez])
    debug && push!(boundarycuts, cutgrid)
  end

  debug ? (realizations, boundarycuts, voxelreusage) : realizations
end

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

function convdist(Xs::AbstractArray, masks::AbstractArray; weights=nothing, inner=true)
  padding = inner == true ? "inner" : "symmetric"

  result = []
  for (X, mask) in zip(Xs, masks)
    weights = weights ≠ nothing ? weights : ones(mask)

    A² = imfilter_fft(X.^2, weights.*ones(mask), padding)
    AB = imfilter_fft(X, weights.*mask, padding)
    B² = sum((weights.*mask).^2)

    push!(result, abs(A² - 2AB + B²))
  end

  sum(result)
end

function overlapdist(X₁::AbstractArray, X₂::AbstractArray, nvertices::Integer)
  # if not categorical, return Euclidean distance
  nvertices == -1 && return (X₁ - X₂).^2

  simplex₁ = simplex_transform(X₁, nvertices)
  simplex₂ = simplex_transform(X₂, nvertices)

  result = []
  for (X, Y) in zip(simplex₁, simplex₂)
    push!(result, (X - Y).^2)
  end

  sum(result)
end

function boundary_cut(overlap::AbstractArray, dir::Symbol)
  # permute overlap cube dimensions so that the algorithm
  # is the same for cuts in x, y and z directions.
  B = []
  if dir == :x
    B = permutedims(overlap, [1,2,3])
  elseif dir == :y
    B = permutedims(overlap, [2,1,3])
  elseif dir == :z
    B = permutedims(overlap, [3,2,1])
  end

  mx, my, mz = size(B)

  # accumulation cube and overlap mask
  E = zeros(B); M = falses(B)

  # pad accumulation cube with +inf
  Epad = (i,j,k) -> all(0 .< [i,j,k] .≤ [mx,my,mz]) ? E[i,j,k] : Inf

  # forward accumulation along 3D cube
  E[:,:,1] = B[:,:,1]
  for k=2:mz, i=1:mx, j=1:my
    square = [Epad(i-1 , j-1 , k-1) , Epad(i-1 , j , k-1) , Epad(i-1 , j+1 , k-1),
              Epad(i   , j-1 , k-1) , Epad(i   , j , k-1) , Epad(i   , j+1 , k-1),
              Epad(i+1 , j-1 , k-1) , Epad(i+1 , j , k-1) , Epad(i+1 , j+1 , k-1)]
    E[i,j,k] = B[i,j,k] + minimum(square)
  end

  # forward accumulation along 2D square (last slice of the cube)
  zslice = slice(E,:,:,mz)
  for j=2:my
    zslice[1,j] += minimum(zslice[1:2,j-1])
    for i=2:mx-1
      zslice[i,j] += minimum(zslice[i-1:i+1,j-1])
    end
    zslice[mx,j] += minimum(zslice[mx-1:mx,j-1])
  end

  # backward search along last slice
  _, idx = findmin(zslice[:,my])
  mslice = slice(M,:,:,mz)
  mslice[1:idx,my] = trues(idx)
  idxvec = zeros(Int, my); idxvec[my] = idx # keep track of indexes
  for j=my-1:-1:1
    for i=1:mx
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
    yslice = slice(E,:,j,:)
    idx = idxvec[j]
    for k=mz-1:-1:1
      for i=1:mx
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
