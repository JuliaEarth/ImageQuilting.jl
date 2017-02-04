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
               soft=nothing, hard=nothing, tol=.1,
               cut=:boykov, path=:rasterup, simplex=false, nreal=1,
               threads=CPU_PHYSICAL_CORES, gpu=false, debug=false, showprogress=false)

  # GPU setup
  global GPU = gpu ? gpu_setup() : nothing

  # use all CPU cores in FFT
  FFTW.set_num_threads(threads)

  # sanity checks
  @assert ndims(training_image) == 3 "training image is not 3D (add ghost dimension for 2D)"
  @assert all(0 .< [tplsizex, tplsizey, tplsizez] .≤ [size(training_image)...]) "invalid template size"
  @assert all([gridsizex, gridsizey, gridsizez] .≥ [tplsizex, tplsizey, tplsizez]) "invalid grid size"
  @assert all(0 .< [overlapx, overlapy, overlapz] .< 1) "overlaps must be in range (0,1)"
  @assert 0 < tol ≤ 1 "tolerance must be in range (0,1]"
  @assert cut ∈ [:dijkstra,:boykov] "invalid cut algorithm"
  @assert path ∈ [:rasterup,:rasterdown,:dilation,:random] "invalid simulation path"
  @assert nreal > 0 "invalid number of realizations"

  # GPU checks
  if gpu
    @assert all([size(training_image)...] .> 1) "GPU support for 3D training images only"
  end

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

  # warn in case of 1-voxel overlaps
  if any(([tplsizex, tplsizey, tplsizez] .>  1) &
         ([overlapx, overlapy, overlapz] .== 1))
    warn("Overlaps with only 1 voxel. Check template/overlap configuration.")
  end

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
  skipped = Set(); datum = []; rastered = []
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
      else
        rastered[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = true
        if any(preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
          push!(datum, (i,j,k))
        end
      end
    end
    rastered[!activated] = false

    # grid must contain active voxels
    any_activated = any(activated[1:gridsizex,1:gridsizey,1:gridsizez])
    @assert any_activated "simulation grid has no active voxel"
  end

  # always work with floating point
  TI = map(Float64, training_image)

  # inactive voxels in the training image
  NaNTI = isnan(TI); TI[NaNTI] = 0

  # perform simplex transform
  simplexTI = [TI]; nvertices = 1
  if simplex
    categories = Set(TI[!NaNTI])
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

      auxpad = padarray(aux.data, Pad(:symmetric, [0,0,0], [nx-lx,ny-ly,nz-lz]))
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

  # main output is a vector of 3D grids
  realizations = []

  # for each realization we have:
  boundarycuts = [] # boundary cut
  voxelreuse = Float64[] # voxel reuse

  # show progress and estimated time duration
  showprogress && (progress = Progress(nreal, color=:black))

  # preallocate memory for distance calculations
  distance = Array(Float64, mₜ-tplsizex+1, nₜ-tplsizey+1, pₜ-tplsizez+1)

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
      simdev = view(simgrid, iₛ:iₑ,jₛ:jₑ,kₛ:kₑ)

      # compute overlap distance
      distance[:] = 0
      if overlapx > 1 && (i-1,j,k) ∈ pasted
        ovx = view(simdev,1:overlapx,:,:)
        xsimplex = simplex ? simplex_transform(ovx, nvertices) : [ovx]

        D = convdist(simplexTI, xsimplex)
        broadcast!(+, distance, distance, view(D,1:mₜ-tplsizex+1,:,:))
      end
      if overlapx > 1 && (i+1,j,k) ∈ pasted
        ovx = view(simdev,spacingx+1:tplsizex,:,:)
        xsimplex = simplex ? simplex_transform(ovx, nvertices) : [ovx]

        D = convdist(simplexTI, xsimplex)
        broadcast!(+, distance, distance, view(D,spacingx+1:mₜ-overlapx+1,:,:))
      end
      if overlapy > 1 && (i,j-1,k) ∈ pasted
        ovy = view(simdev,:,1:overlapy,:)
        ysimplex = simplex ? simplex_transform(ovy, nvertices) : [ovy]

        D = convdist(simplexTI, ysimplex)
        broadcast!(+, distance, distance, view(D,:,1:nₜ-tplsizey+1,:))
      end
      if overlapy > 1 && (i,j+1,k) ∈ pasted
        ovy = view(simdev,:,spacingy+1:tplsizey,:)
        ysimplex = simplex ? simplex_transform(ovy, nvertices) : [ovy]

        D = convdist(simplexTI, ysimplex)
        broadcast!(+, distance, distance, view(D,:,spacingy+1:nₜ-overlapy+1,:))
      end
      if overlapz > 1 && (i,j,k-1) ∈ pasted
        ovz = view(simdev,:,:,1:overlapz)
        zsimplex = simplex ? simplex_transform(ovz, nvertices) : [ovz]

        D = convdist(simplexTI, zsimplex)
        broadcast!(+, distance, distance, view(D,:,:,1:pₜ-tplsizez+1))
      end
      if overlapz > 1 && (i,j,k+1) ∈ pasted
        ovz = view(simdev,:,:,spacingz+1:tplsizez)
        zsimplex = simplex ? simplex_transform(ovz, nvertices) : [ovz]

        D = convdist(simplexTI, zsimplex)
        broadcast!(+, distance, distance, view(D,:,:,spacingz+1:pₜ-overlapz+1))
      end

      # disable dataevents that contain inactive voxels
      distance[disabled] = Inf

      # compute hard and soft distances
      auxdistances = []
      if hard ≠ nothing && any(preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
        harddev = hardgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        hsimplex = simplex ? simplex_transform(harddev, nvertices) : [harddev]
        D = convdist(simplexTI, hsimplex, weights=preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])

        # disable dataevents that contain inactive voxels
        D[disabled] = Inf

        # swap overlap and hard distances
        push!(auxdistances, distance)
        distance = D
      end
      for n=1:length(softTI)
        softdev = softgrid[n][iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        D = convdist([softTI[n]], [softdev])

        # disable dataevents that contain inactive voxels
        D[disabled] = Inf

        push!(auxdistances, D)
      end

      # current pattern database
      patterndb = isempty(auxdistances) ? find(distance .≤ (1+tol)minimum(distance)) :
                                          relaxation(distance, auxdistances, tol)
      patternprobs = tau_model(patterndb, distance, auxdistances)

      # pick a pattern at random from the database
      idx = sample(patterndb, weights(patternprobs))
      iᵦ, jᵦ, kᵦ = ind2sub(size(distance), idx)

      # selected training image dataevent
      TIdev = view(TI,iᵦ:iᵦ+tplsizex-1,jᵦ:jᵦ+tplsizey-1,kᵦ:kᵦ+tplsizez-1)

      # boundary cut mask
      M = falses(simdev)
      if overlapx > 1 && (i-1,j,k) ∈ pasted
        A = view(simdev,1:overlapx,:,:); B = view(TIdev,1:overlapx,:,:)
        M[1:overlapx,:,:] |= boundary_cut(A, B, :x)
      end
      if overlapx > 1 && (i+1,j,k) ∈ pasted
        A = view(simdev,spacingx+1:tplsizex,:,:); B = view(TIdev,spacingx+1:tplsizex,:,:)
        M[spacingx+1:tplsizex,:,:] |= flipdim(boundary_cut(flipdim(A, 1), flipdim(B, 1), :x), 1)
      end
      if overlapy > 1 && (i,j-1,k) ∈ pasted
        A = view(simdev,:,1:overlapy,:); B = view(TIdev,:,1:overlapy,:)
        M[:,1:overlapy,:] |= boundary_cut(A, B, :y)
      end
      if overlapy > 1 && (i,j+1,k) ∈ pasted
        A = view(simdev,:,spacingy+1:tplsizey,:); B = view(TIdev,:,spacingy+1:tplsizey,:)
        M[:,spacingy+1:tplsizey,:] |= flipdim(boundary_cut(flipdim(A, 2), flipdim(B, 2), :y), 2)
      end
      if overlapz > 1 && (i,j,k-1) ∈ pasted
        A = view(simdev,:,:,1:overlapz); B = view(TIdev,:,:,1:overlapz)
        M[:,:,1:overlapz] |= boundary_cut(A, B, :z)
      end
      if overlapz > 1 && (i,j,k+1) ∈ pasted
        A = view(simdev,:,:,spacingz+1:tplsizez); B = view(TIdev,:,:,spacingz+1:tplsizez)
        M[:,:,spacingz+1:tplsizez] |= flipdim(boundary_cut(flipdim(A, 3), flipdim(B, 3), :z), 3)
      end

      # paste quilted pattern from training image
      simdev[!M] = TIdev[!M]

      # save boundary cut
      debug && (cutgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M)
    end

    # save voxel reuse
    debug && push!(voxelreuse, sum(cutgrid)/overlap_volume)

    # hard data and shape correction
    if hard ≠ nothing
      simgrid[preset] = hardgrid[preset]
      simgrid[!activated] = NaN
      debug && (cutgrid[!activated] = NaN)
    end

    # throw away voxels that are outside of the grid
    simgrid = view(simgrid,1:gridsizex,1:gridsizey,1:gridsizez)
    debug && (cutgrid = view(cutgrid,1:gridsizex,1:gridsizey,1:gridsizez))

    # save and continue
    push!(realizations, simgrid)
    debug && push!(boundarycuts, cutgrid)

    # update progress bar
    showprogress && next!(progress)
  end

  debug ? (realizations, boundarycuts, voxelreuse) : realizations
end
