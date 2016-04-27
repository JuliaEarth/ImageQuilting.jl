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

function iqsim(training_image::AbstractArray,
               tplsizex::Integer, tplsizey::Integer, tplsizez::Integer,
               gridsizex::Integer, gridsizey::Integer, gridsizez::Integer;
               overlapx=1/6, overlapy=1/6, overlapz=1/6,
               soft=nothing, hard=nothing, cutoff=.1,
               cut=:dijkstra, path=:rasterup, categorical=false,
               seed=0, nreal=1, debug=false)

  # sanity checks
  @assert ndims(training_image) == 3 "training image is not 3D (add ghost dimension for 2D)"
  @assert all(0 .< [tplsizex, tplsizey, tplsizez] .≤ [size(training_image)...]) "invalid template size"
  @assert all([gridsizex, gridsizey, gridsizez] .≥ [tplsizex, tplsizey, tplsizez]) "invalid grid size"
  @assert all(0 .< [overlapx, overlapy, overlapz] .< 1) "overlaps must be in range (0,1)"
  @assert 0 < cutoff ≤ 1 "cutoff must be in range (0,1]"
  @assert cut ∈ [:dijkstra,:boykov] "invalid cut algorithm"
  @assert path ∈ [:rasterup,:rasterdown,:dilation,:random] "invalid simulation path"

  # soft data checks
  if soft ≠ nothing
    @assert isa(soft, SoftData) || isa(soft, AbstractArray{SoftData})

    # encapsulate single auxiliary variable in an array
    isa(soft, SoftData) && (soft = [soft])

    for aux in soft
      @assert ndims(aux.data) == 3 "soft data is not 3D (add ghost dimension for 2D)"
      @assert all([size(aux.data)...] .≥ [gridsizex, gridsizey, gridsizez]) "soft data size < grid size"
    end
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

  # keep track of hard data and inactive voxels
  skipped = Set(); datum = Set(); rastered = []
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

      if all(!activated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
        push!(skipped, (i,j,k))
      elseif all(!preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
        rastered[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = true
      else
        push!(datum, (i,j,k))
      end
    end
    rastered[!activated] = false

    simulated = preset | rastered

    # path must be non-empty or data must be available
    any_activated = any(activated[1:gridsizex,1:gridsizey,1:gridsizez])
    any_simulated = any(simulated[1:gridsizex,1:gridsizey,1:gridsizez])
    @assert any_activated "simulation grid has no active voxel"
    @assert any_simulated "path must visit at least one tile in the absence of data"
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

  # pad soft data and soft transform training image
  softgrid = []; softTI = []
  if soft ≠ nothing
    for aux in soft
      mx, my, mz = size(aux.data)
      lx = min(mx,nx); ly = min(my,ny); lz = min(mz,nz)

      auxpad = padarray(aux.data, [0,0,0], [nx-lx,ny-ly,nz-lz], "symmetric")
      auxpad[isnan(auxpad)] = 0

      push!(softgrid, auxpad)

      auxTI = copy(aux.transform(training_image))

      @assert size(auxTI) == size(TI) "auxiliary TI must have the same size as TI"

      # inactive voxels in the auxiliary training image
      auxTI[NaNTI] = 0

      push!(softTI, auxTI)
    end
  end

  # overwrite path option if data is available
  !isempty(datum) && (path = :datum)

  # select cut algorithm
  boundary_cut = cut == :dijkstra ? dijkstra_cut : boykov_kolmogorov_cut

  # use all CPU cores in FFT
  FFTW.set_num_threads(CPU_CORES)

  # main output is a vector of 3D grids
  realizations = []

  # for each realization we have:
  boundarycuts = [] # boundary cut
  voxelreuse = [] # voxel reuse

  # set seed and start
  srand(seed)

  for real=1:nreal
    # allocate memory for current simulation
    simgrid = zeros(nx, ny, nz)
    cutgrid = debug ? zeros(nx, ny, nz) : []

    # keep track of pasted tiles
    pasted = Set()

    # construct simulation path
    simpath = genpath((ntilex,ntiley,ntilez), path, datum)

    # loop simulation grid tile by tile
    for pathidx in simpath
      i, j, k = ind2sub((ntilex,ntiley,ntilez), pathidx)

      # skip tile if all voxels are inactive
      (i,j,k) ∈ skipped && continue

      # if not skipped, proceed and paste tile
      push!(pasted, (i,j,k))

      # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
      iₛ = (i-1)spacingx + 1
      jₛ = (j-1)spacingy + 1
      kₛ = (k-1)spacingz + 1
      iₑ = iₛ + tplsizex - 1
      jₑ = jₛ + tplsizey - 1
      kₑ = kₛ + tplsizez - 1

      # current simulation dataevent
      simdev = simgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]

      # compute overlap distance
      distance = zeros(mₜ-tplsizex+1, nₜ-tplsizey+1, pₜ-tplsizez+1)
      if overlapx > 1 && (i-1,j,k) ∈ pasted
        ovx = simdev[1:overlapx,:,:]
        xsimplex = categorical ? simplex_transform(ovx, nvertices) : Any[ovx]

        D = convdist(simplexTI, xsimplex)
        distance += D[1:mₜ-tplsizex+1,:,:]
      end
      if overlapx > 1 && (i+1,j,k) ∈ pasted
        ovx = simdev[spacingx+1:end,:,:]
        xsimplex = categorical ? simplex_transform(ovx, nvertices) : Any[ovx]

        D = convdist(simplexTI, xsimplex)
        distance += D[spacingx+1:end,:,:]
      end
      if overlapy > 1 && (i,j-1,k) ∈ pasted
        ovy = simdev[:,1:overlapy,:]
        ysimplex = categorical ? simplex_transform(ovy, nvertices) : Any[ovy]

        D = convdist(simplexTI, ysimplex)
        distance += D[:,1:nₜ-tplsizey+1,:]
      end
      if overlapy > 1 && (i,j+1,k) ∈ pasted
        ovy = simdev[:,spacingy+1:end,:]
        ysimplex = categorical ? simplex_transform(ovy, nvertices) : Any[ovy]

        D = convdist(simplexTI, ysimplex)
        distance += D[:,spacingy+1:end,:]
      end
      if overlapz > 1 && (i,j,k-1) ∈ pasted
        ovz = simdev[:,:,1:overlapz]
        zsimplex = categorical ? simplex_transform(ovz, nvertices) : Any[ovz]

        D = convdist(simplexTI, zsimplex)
        distance += D[:,:,1:pₜ-tplsizez+1]
      end
      if overlapz > 1 && (i,j,k+1) ∈ pasted
        ovz = simdev[:,:,spacingz+1:end]
        zsimplex = categorical ? simplex_transform(ovz, nvertices) : Any[ovz]

        D = convdist(simplexTI, zsimplex)
        distance += D[:,:,spacingz+1:end]
      end

      # disable dataevents that contain inactive voxels
      distance[disabled] = Inf

      # compute soft and hard distances
      auxdistances = []
      for n=1:length(softTI)
        softdev = softgrid[n][iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        D = convdist(Any[softTI[n]], Any[softdev])

        # disable dataevents that contain inactive voxels
        D[disabled] = Inf

        push!(auxdistances, D)
      end
      if hard ≠ nothing && any(preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
        harddev = hardgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        hsimplex = categorical ? simplex_transform(harddev, nvertices) : Any[harddev]
        D = convdist(simplexTI, hsimplex, weights=preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])

        # disable dataevents that contain inactive voxels
        D[disabled] = Inf

        push!(auxdistances, D)
      end

      # current pattern database
      patterndb = isempty(auxdistances) ? find(distance .≤ (1+cutoff)minimum(distance)) :
                                          relaxation(distance, auxdistances, cutoff)
      patternprobs = tau_model(patterndb, distance, auxdistances)

      # pick a pattern at random from the database
      idx = sample(patterndb, weights(patternprobs))
      iᵦ, jᵦ, kᵦ = ind2sub(size(distance), idx)

      # selected training image dataevent
      TIdev = TI[iᵦ:iᵦ+tplsizex-1,jᵦ:jᵦ+tplsizey-1,kᵦ:kᵦ+tplsizez-1]

      # boundary cut mask
      M = falses(simdev)
      if overlapx > 1 && (i-1,j,k) ∈ pasted
        A = simdev[1:overlapx,:,:]; B = TIdev[1:overlapx,:,:]
        M[1:overlapx,:,:] |= boundary_cut(A, B, :x)
      end
      if overlapx > 1 && (i+1,j,k) ∈ pasted
        A = simdev[spacingx+1:end,:,:]; B = TIdev[spacingx+1:end,:,:]
        M[spacingx+1:end,:,:] |= flipdim(boundary_cut(flipdim(A, 1), flipdim(B, 1), :x), 1)
      end
      if overlapy > 1 && (i,j-1,k) ∈ pasted
        A = simdev[:,1:overlapy,:]; B = TIdev[:,1:overlapy,:]
        M[:,1:overlapy,:] |= boundary_cut(A, B, :y)
      end
      if overlapy > 1 && (i,j+1,k) ∈ pasted
        A = simdev[:,spacingy+1:end,:]; B = TIdev[:,spacingy+1:end,:]
        M[:,spacingy+1:end,:] |= flipdim(boundary_cut(flipdim(A, 2), flipdim(B, 2), :y), 2)
      end
      if overlapz > 1 && (i,j,k-1) ∈ pasted
        A = simdev[:,:,1:overlapz]; B = TIdev[:,:,1:overlapz]
        M[:,:,1:overlapz] |= boundary_cut(A, B, :z)
      end
      if overlapz > 1 && (i,j,k+1) ∈ pasted
        A = simdev[:,:,spacingz+1:end]; B = TIdev[:,:,spacingz+1:end]
        M[:,:,spacingz+1:end] |= flipdim(boundary_cut(flipdim(A, 3), flipdim(B, 3), :z), 3)
      end

      # paste contributions from simulation grid and training image
      simgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M.*simdev + !M.*TIdev

      # save boundary cut
      debug && (cutgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M)
    end

    # prepare tiles with hard data for dilation
    if hard ≠ nothing
      simgrid[!rastered] = 0
      simgrid[preset] = hardgrid[preset]
    end

    # throw away voxels that are outside of the grid
    simgrid = simgrid[1:gridsizex,1:gridsizey,1:gridsizez]

    # do the same for boundary cut, but only after saving voxel reuse
    debug && push!(voxelreuse, sum(cutgrid)/overlap_volume)
    debug && (cutgrid = cutgrid[1:gridsizex,1:gridsizey,1:gridsizez])

    #-----------------------------------------------------------------

    # simulate remaining voxels
    if hard ≠ nothing
      tplx, tply, tplz = tplsizex, tplsizey, tplsizez

      simulated = preset | rastered

      # throw away voxels that are outside of the grid
      simulated = simulated[1:gridsizex,1:gridsizey,1:gridsizez]
      activated = activated[1:gridsizex,1:gridsizey,1:gridsizez]

      # simulation frontier
      dilated = dilate(simulated, [1,2,3]) & activated
      frontier = find(dilated - simulated)

      # confidence map
      C = map(Float64, simulated)

      while !isempty(frontier)
        visited = 0

        # update confidence in the frontier
        for vox in frontier
          # tile center is given by (iᵥ,jᵥ,kᵥ)
          iᵥ, jᵥ, kᵥ = ind2sub(size(simgrid), vox)

          # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
          iₛ = max(iᵥ - (tplx-1)÷2, 1)
          jₛ = max(jᵥ - (tply-1)÷2, 1)
          kₛ = max(kᵥ - (tplz-1)÷2, 1)
          iₑ = min(iᵥ + tplx÷2, gridsizex)
          jₑ = min(jᵥ + tply÷2, gridsizey)
          kₑ = min(kᵥ + tplz÷2, gridsizez)

          confdev = C[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
          booldev = simulated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]

          C[vox] = sum(confdev[booldev]) / (tplx*tply*tplz)
        end

        # data-driven visiting order
        permvec = sortperm(C[frontier], rev=true)
        frontier = frontier[permvec]

        if any([tplx,tply,tplz] .> 1)
          # disable tiles in the training image if they contain inactive voxels
          disabledₜ = copy(NaNTI)
          for i=1:tplx÷2
            disabledₜ = dilate(disabledₜ, [1])
          end
          for j=1:tply÷2
            disabledₜ = dilate(disabledₜ, [2])
          end
          for k=1:tplz÷2
            disabledₜ = dilate(disabledₜ, [3])
          end

          # scan training image
          for vox in frontier
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

              # compute soft and hard distances
              auxdistances = []
              for n=1:length(softTI)
                softdev = softgrid[n][iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
                D = convdist(Any[softTI[n]], Any[softdev], weights=booldev, inner=false)

                # disable dataevents that contain inactive voxels
                D[disabledₜ] = Inf

                push!(auxdistances, D)
              end
              if hard ≠ nothing && any(preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
                harddev = hardgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
                hsimplex = categorical ? simplex_transform(harddev, nvertices) : Any[harddev]
                D = convdist(simplexTI, hsimplex, weights=preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ], inner=false)

                # disable dataevents that contain inactive voxels
                D[disabledₜ] = Inf

                push!(auxdistances, D)
              end

              # current pattern database
              patterndb = isempty(auxdistances) ? find(distance .≤ (1+cutoff)minimum(distance)) :
                                                  relaxation(distance, auxdistances, cutoff)
              patternprobs = tau_model(patterndb, distance, auxdistances)

              # pick a pattern at random from the database
              idx = sample(patterndb, weights(patternprobs))
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
          for vox in frontier
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

        dilated = dilate(simulated, [1,2,3]) & activated
        frontier = find(dilated - simulated)
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
