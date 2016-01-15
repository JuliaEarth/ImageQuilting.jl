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

if VERSION > v"0.5-"
  using Combinatorics: combinations
end

include("datatypes.jl")
include("boundary_cut.jl")
include("simplex_transform.jl")

function iqsim(training_image::AbstractArray,
               tplsizex::Integer, tplsizey::Integer, tplsizez::Integer,
               gridsizex::Integer, gridsizey::Integer, gridsizez::Integer;
               overlapx=1/6, overlapy=1/6, overlapz=1/6,
               soft=nothing, hard=nothing, cutoff=.1, softcutoff=.1,
               seed=0, nreal=1, categorical=false, debug=false)

  # sanity checks
  @assert ndims(training_image) == 3 "training image is not 3D (add ghost dimension for 2D)"
  @assert all(0 .< [tplsizex, tplsizey, tplsizez] .≤ [size(training_image)...]) "invalid template size"
  @assert all([gridsizex, gridsizey, gridsizez] .≥ [tplsizex, tplsizey, tplsizez]) "invalid grid size"
  @assert all(0 .< [overlapx, overlapy, overlapz] .< 1) "overlaps must be in range (0,1)"
  @assert cutoff > 0 "cutoff must be positive"

  # soft data checks
  if soft ≠ nothing
    @assert isa(soft, SoftData) || isa(soft, AbstractArray{SoftData})

    # encapsulate single auxiliary variable in an array
    isa(soft, SoftData) && (soft = [soft])

    for aux in soft
      @assert ndims(aux.data) == 3 "soft data is not 3D (add ghost dimension for 2D)"
      @assert all([size(aux.data)...] .≥ [gridsizex, gridsizey, gridsizez]) "soft data size < grid size"
    end

    @assert 0 < cutoff ≤ 1 "cutoff must be in range (0,1] when soft data is available"
    @assert 0 < softcutoff ≤ 1 "softcutoff must be in range (0,1]"
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

  # simulation grid dimensions
  nx = ntilex * (tplsizex - overlapx) + overlapx
  ny = ntiley * (tplsizey - overlapy) + overlapy
  nz = ntilez * (tplsizez - overlapz) + overlapz

  # total overlap volume in simulation grid
  overlap_volume = nx*ny*nz - (nx - (ntilex-1)overlapx)*
                              (ny - (ntiley-1)overlapy)*
                              (nz - (ntilez-1)overlapz)

  # hard data in grid format
  hardgrid = []; preset = []; activated = []
  if hard ≠ nothing
    hardgrid = zeros(nx, ny, nz)
    preset = falses(nx, ny, nz)
    activated = trues(nx, ny, nz)
    for loc in keys(hard)
      if isnan(hard[loc])
        activated[loc...] = false
      else
        hardgrid[loc...] = hard[loc]
        preset[loc...] = true
      end
    end
    activated[gridsizex+1:nx,:,:] = false
    activated[:,gridsizey+1:ny,:] = false
    activated[:,:,gridsizez+1:nz] = false
  end

  # tiles that contain hard data are skipped during raster path
  skipped = Set(); rastered = []
  if hard ≠ nothing
    rastered = falses(nx, ny, nz)
    for k=1:ntilez, j=1:ntiley, i=1:ntilex
      # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
      iₛ = (i-1)spacingx + 1
      jₛ = (j-1)spacingy + 1
      kₛ = (k-1)spacingz + 1
      iₑ = iₛ + tplsizex - 1
      jₑ = jₛ + tplsizey - 1
      kₑ = kₛ + tplsizez - 1

      if any(preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]) || all(!activated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
        push!(skipped, (i,j,k))
      else
        rastered[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = true
      end
    end
    rastered[!activated] = false
  end

  # raster path must be non-empty or data must be available
  if hard ≠ nothing
    simulated = preset | rastered
    any_activated = any(activated[1:gridsizex,1:gridsizey,1:gridsizez])
    any_simulated = any(simulated[1:gridsizex,1:gridsizey,1:gridsizez])
    @assert any_activated "simulation grid has no active voxel"
    @assert any_simulated "raster path must visit at least one tile in the absence of data"
  end

  # always work with floating point
  TI = map(Float64, training_image)

  # inactive voxels in the training image
  NaNTI = isnan(training_image); TI[NaNTI] = 0

  # perform simplex transform
  simplexTI = Any[TI]; nvertices = 1
  if categorical
    categories = Set(training_image[!NaNTI])
    ncategories = nvertices = length(categories) - 1

    @assert categories == Set(0:ncategories) "categories should be labeled 1, 2, 3,..."

    simplexTI = simplex_transform(TI, nvertices)
  end

  # pad soft data and soft transform training image
  softgrid = []; softTI = []
  if soft ≠ nothing
    for aux in soft
      mx, my, mz = size(aux.data)
      lx = min(mx,nx); ly = min(my,ny); lz = min(mz,nz)

      auxpad = padarray(aux.data, [0,0,0], [nx-lx,ny-ly,nz-lz], "symmetric")
      auxpad[isnan(auxpad)] = 0

      push!(softgrid, auxpad)

      auxTI = copy(aux.transform(TI))

      @assert size(auxTI) == size(TI) "auxiliary TI must have the same size as TI"

      # inactive voxels in the auxiliary training image
      auxTI[NaNTI] = 0

      push!(softTI, auxTI)
    end
  end

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

  # main output is a vector of 3D grids
  realizations = []

  # for each realization we have:
  boundarycuts = [] # boundary cut
  voxelreuse = [] # voxel reuse

  # set seed and start
  srand(seed)

  # use all CPU cores in FFT
  FFTW.set_num_threads(CPU_CORES)

  for real=1:nreal
    # allocate memory for current simulation
    simgrid = zeros(nx, ny, nz)
    cutgrid = debug ? zeros(nx, ny, nz) : []

    # preset hard data
    simgrid[preset] = hardgrid[preset]

    # loop simulation grid tile by tile
    for k=1:ntilez, j=1:ntiley, i=1:ntilex
      # skip tile if it contains hard data
      (i,j,k) ∈ skipped && continue

      # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
      iₛ = (i-1)spacingx + 1
      jₛ = (j-1)spacingy + 1
      kₛ = (k-1)spacingz + 1
      iₑ = iₛ + tplsizex - 1
      jₑ = jₛ + tplsizey - 1
      kₑ = kₛ + tplsizez - 1

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
        softdistance = []
        for n=1:length(soft)
          # compute the distance between the soft dataevent and
          # all dataevents in the soft training image
          softdev = softgrid[n][iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
          D = convdist(Any[softTI[n]], Any[softdev])

          # disable dataevents that contain inactive voxels
          D[disabled] = Inf

          push!(softdistance, D)
        end

        # candidates with good overlap
        dbsize = ceil(Int, cutoff*length(distance))
        overlapdb = (i==1 && j==1 && k==1) ? collect(1:length(distance)) :
                                             sortperm(distance[:])[1:dbsize]

        # candidates in accordance with soft data
        softdbs = [sortperm(softdistance[n][:]) for n=1:length(soft)]

        patterndb = relaxation(overlapdb, softdbs, softcutoff, length(distance))
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

      # save boundary cut
      debug && (cutgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M)
    end

    # throw away voxels that are outside of the grid
    simgrid = simgrid[1:gridsizex,1:gridsizey,1:gridsizez]

    # do the same for boundary cut, but only after saving voxel reuse
    debug && push!(voxelreuse, sum(cutgrid)/overlap_volume)
    debug && (cutgrid = cutgrid[1:gridsizex,1:gridsizey,1:gridsizez])

    #-----------------------------------------------------------------

    # simulate remaining voxels skipped during raster path
    if hard ≠ nothing
      tplx, tply, tplz = tplsizex, tplsizey, tplsizez

      simulated = preset | rastered

      # throw away voxels that are outside of the grid
      simulated = simulated[1:gridsizex,1:gridsizey,1:gridsizez]
      activated = activated[1:gridsizex,1:gridsizey,1:gridsizez]

      # morphological dilation
      dilated = dilate(simulated) & activated

      while dilated ≠ simulated
        visited = 0

        # disable tiles in the training image if they contain inactive voxels
        disabledₜ = copy(NaNTI)
        for i=1:tplx-1
          disabledₜ = dilate(disabledₜ, [1])
        end
        for j=1:tply-1
          disabledₜ = dilate(disabledₜ, [2])
        end
        for k=1:tplz-1
          disabledₜ = dilate(disabledₜ, [3])
        end

        if any([tplx,tply,tplz] .> 1)
          # scan training image
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
                softdistance = []
                for n=1:length(soft)
                  # compute the distance between the soft dataevent and
                  # all dataevents in the soft training image
                  softdev = softgrid[n][iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
                  D = convdist(Any[softTI[n]], Any[softdev], weights=booldev, inner=false)

                  # disable dataevents that contain inactive voxels
                  D[disabledₜ] = Inf

                  push!(softdistance, D)
                end

                # candidates with good overlap
                dbsize = ceil(Int, cutoff*length(distance))
                overlapdb = sortperm(distance[:])[1:dbsize]

                # candidates in accordance with soft data
                softdbs = [sortperm(softdistance[n][:]) for n=1:length(soft)]

                patterndb = relaxation(overlapdb, softdbs, softcutoff, length(distance))
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
                if all(0 .< [i,j,k] .≤ [size(TI)...]) && !NaNTI[i,j,k]
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
        else
          # copy a neighbor voxel from simulation grid
          for vox in find(dilated - simulated)
            # mark location as visited
            visited += 1

            # tile center is given by (iᵥ,jᵥ,kᵥ)
            iᵥ, jᵥ, kᵥ = ind2sub(size(simgrid), vox)

            # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
            iₛ, jₛ, kₛ = max(iᵥ-1, 1), max(jᵥ-1, 1), max(kᵥ-1, 1)
            iₑ, jₑ, kₑ = min(iᵥ+1, gridsizex), min(jᵥ+1, gridsizey), min(kᵥ+1, gridsizez)

            # voxel-centered dataevent
            simdev = simgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]

            # pick a voxel in the simulated neighborhood at random
            neighborhood = find(activated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] & simulated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
            idx = neighborhood[rand(1:length(neighborhood))]

            # copy it
            simgrid[iᵥ,jᵥ,kᵥ] = simdev[idx]
            simulated[iᵥ,jᵥ,kᵥ] = true
          end
        end

        if visited == 0
          # reduce the template size and proceed
          idx = indmax([tplz,tply,tplx])
          idx == 3 && (tplx = max(tplx-1,1))
          idx == 2 && (tply = max(tply-1,1))
          idx == 1 && (tplz = max(tplz-1,1))
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

  debug ? (realizations, boundarycuts, voxelreuse) : realizations
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

function quick_intersect(A::Vector{Int}, B::Vector{Int}, nbits::Integer)
  bitsA = falses(nbits)
  bitsB = falses(nbits)
  bitsA[A] = true
  bitsB[B] = true

  find(bitsA & bitsB)
end

function relaxation(overlapdb::AbstractVector, softdbs::AbstractVector,
                    initcutoff::Real, npatterns::Integer)
  τₛ = initcutoff
  patterndb = []
  while true
    softdbsize = ceil(Int, τₛ*npatterns)

    patterndb = overlapdb
    for n=1:length(softdbs)
      softdb = softdbs[n][1:softdbsize]
      patterndb = quick_intersect(patterndb, softdb, npatterns)

      isempty(patterndb) && break
    end

    !isempty(patterndb) && break
    τₛ = min(τₛ + .1, 1)
  end

  patterndb
end
