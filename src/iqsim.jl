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

include("datatypes.jl")
include("boundary_cut.jl")
include("simplex_transform.jl")

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

  # inactive voxels in the training image
  NaNTI = isnan(training_image)
  TI[NaNTI] = 0

  # disable tiles in the training image if they contain inactive voxels
  mₜ, nₜ, pₜ = size(TI)
  disabled = falses(mₜ-tplsizex+1, nₜ-tplsizey+1, pₜ-tplsizez+1)
  for nanidx in find(NaNTI)
    iₙ, jₙ, kₙ = ind2sub(size(TI), nanidx)

    # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
    iₛ = max(iₙ-tplsizex+1, 1)
    jₛ = max(jₙ-tplsizey+1, 1)
    kₛ = max(kₙ-tplsizez+1, 1)
    iₑ = min(iₙ, mₜ-tplsizex+1)
    jₑ = min(jₙ, nₜ-tplsizey+1)
    kₑ = min(kₙ, pₜ-tplsizez+1)

    disabled[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = true
  end

  # perform simplex transform
  simplexTI = Any[TI]; nvertices = 1
  if categorical
    categories = Set(training_image[!NaNTI])
    ncategories = nvertices = length(categories) - 1

    @assert categories == Set(0:ncategories) "Categories should be labeled 1, 2, 3,..."

    simplexTI = simplex_transform(TI, nvertices)
  end

  # simulation grid dimensions
  nx = ntilex * (tplsizex - overlapx) + overlapx
  ny = ntiley * (tplsizey - overlapy) + overlapy
  nz = ntilez * (tplsizez - overlapz) + overlapz

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

  # tiles that contain hard data are skipped during raster path
  skipped = Set()
  if hard ≠ nothing
    for (x,y,z) in keys(hard)
      i = spacingx > 0 ? min((x-1)÷spacingx+1, ntilex) : 1
      j = spacingy > 0 ? min((y-1)÷spacingy+1, ntiley) : 1
      k = spacingz > 0 ? min((z-1)÷spacingz+1, ntilez) : 1

      push!(skipped, (i,j,k))

      inoverlap = falses(3)
      i > 1 && x ≤ ntilex*spacingx && (x-1-overlapx)÷spacingx+1 < i && (inoverlap[1] = true)
      j > 1 && y ≤ ntiley*spacingy && (y-1-overlapy)÷spacingy+1 < j && (inoverlap[2] = true)
      k > 1 && z ≤ ntilez*spacingz && (z-1-overlapz)÷spacingz+1 < k && (inoverlap[3] = true)

      for n=1:3
        for c in combinations(1:3, n)
          if all(inoverlap[c])
            idx = [i,j,k]; idx[c] -= 1
            push!(skipped, (idx...))
          end
        end
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
    cutgrid = debug ? zeros(nx, ny, nz) : []

    # set hard data for current simulation
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
      (i,j,k) ∈ skipped && continue

      # current simulation dataevent
      simdev = simgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]

      # compute the distance between the simulation dataevent
      # and all patterns in the training image
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

      # disable dataevents that contain inactive voxels
      distance[disabled] = Inf

      # current pattern database
      patterndb = []
      if soft ≠ nothing
        # compute the distance between the soft dataevent and
        # all dataevents in the soft training image
        softdev = softgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        softdistance = convdist(Any[softTI], Any[softdev])

        # disable dataevents that contain inactive voxels
        softdistance[disabled] = Inf

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

    # do the same for boundary cut, but only after saving voxel reusage
    debug && push!(voxelreusage, sum(cutgrid)/overlap_volume)
    debug && (cutgrid = cutgrid[1:gridsizex,1:gridsizey,1:gridsizez])

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

        # disable tiles in the training image if they contain inactive voxels
        disabledₜ = falses(mₜ, nₜ, pₜ)
        for nanidx in find(NaNTI)
          iₙ, jₙ, kₙ = ind2sub(size(TI), nanidx)

          # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
          iₛ = max(iₙ - tplx÷2, 1)
          jₛ = max(jₙ - tply÷2, 1)
          kₛ = max(kₙ - tplz÷2, 1)
          iₑ = min(iₛ + tplx - 1, mₜ)
          jₑ = min(jₛ + tply - 1, nₜ)
          kₑ = min(kₛ + tplz - 1, pₜ)

          disabledₜ[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = true
        end

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

            # compute the distance between the simulation dataevent
            # and all patterns in the training image
            distance = convdist(simplexTI, simplexdev, weights=booldev, inner=false)

            # disable dataevents that contain inactive voxels
            distance[disabledₜ] = Inf

            # current pattern database
            patterndb = []
            if soft ≠ nothing
              # compute the distance between the soft dataevent and
              # all dataevents in the soft training image
              softdev = softgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
              softdistance = convdist(Any[softTI], Any[softdev], weights=booldev, inner=false)

              # disable dataevents that contain inactive voxels
              softdistance[disabledₜ] = Inf

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
      simgrid[!activated] = NaN
      debug && (cutgrid[!activated] = NaN)
    end

    push!(realizations, simgrid)
    debug && push!(boundarycuts, cutgrid)
  end

  debug ? (realizations, boundarycuts, voxelreusage) : realizations
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
