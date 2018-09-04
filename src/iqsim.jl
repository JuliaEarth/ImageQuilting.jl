# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    iqsim(trainimg::AbstractArray{T,N}, tilesize::NTuple{N,Int},
          gridsizex::Integer, gridsizey::Integer, gridsizez::Integer;
          overlapx::Real=1/6, overlapy::Real=1/6, overlapz::Real=1/6,
          soft::AbstractVector=[], hard::HardData=HardData(), tol::Real=.1,
          cut::Symbol=:boykov, path::Symbol=:rasterup, simplex::Bool=false,
          nreal::Integer=1, threads::Integer=CPU_PHYSICAL_CORES,
          gpu::Bool=false, debug::Bool=false, showprogress::Bool=false)

Performs image quilting simulation as described in Hoffimann et al. 2017.

## Parameters

### Required

* `trainimg` is any 3D array (add ghost dimension for 2D)
* `tilesize` is the tile size
* `gridsizex`,`gridsizey`,`gridsizez` is the simulation size

### Optional

* `overlapx`,`overlapy`,`overlapz` is the percentage of overlap
* `soft` is a vector of `(data,dataTI)` pairs
* `hard` is an instance of `HardData`
* `tol` is the initial relaxation tolerance in (0,1] (default to .1)
* `cut` is the cut algorithm (`:dijkstra` or `:boykov`)
* `path` is the simulation path (`:rasterup`, `:rasterdown`, `:dilation` or `:random`)
* `simplex` informs whether to apply or not the simplex transform to the image
* `nreal` is the number of realizations
* `threads` is the number of threads for the FFT (default to all CPU cores)
* `gpu` informs whether to use the GPU or the CPU
* `debug` informs whether to export or not the boundary cuts and voxel reuse
* `showprogress` informs whether to show or not estimated time duration

The main output `reals` consists of a list of 3D realizations that can be indexed with
`reals[1], reals[2], ..., reals[nreal]`. If `debug=true`, additional output is generated:

```julia
reals, cuts, voxs = iqsim(..., debug=true)
```

`cuts[i]` is the boundary cut for `reals[i]` and `voxs[i]` is the associated voxel reuse.
"""
function iqsim(trainimg::AbstractArray{T,N}, tilesize::NTuple{N,Int},
               gridsizex::Integer, gridsizey::Integer, gridsizez::Integer;
               overlapx::Real=1/6, overlapy::Real=1/6, overlapz::Real=1/6,
               soft::AbstractVector=[], hard::HardData=HardData(), tol::Real=.1,
               cut::Symbol=:boykov, path::Symbol=:rasterup, simplex::Bool=false,
               nreal::Integer=1, threads::Integer=CPU_PHYSICAL_CORES,
               gpu::Bool=false, debug::Bool=false, showprogress::Bool=false) where {T,N}

  # use all CPU cores in FFT
  # FFTW.set_num_threads(threads)

  # sanity checks
  @assert ndims(trainimg) == 3 "training image is not 3D (add ghost dimension for 2D)"
  @assert all(0 .< tilesize .≤ size(trainimg)) "invalid tile size"
  @assert all((gridsizex, gridsizey, gridsizez) .≥ tilesize) "invalid grid size"
  @assert all(0 .< [overlapx, overlapy, overlapz] .< 1) "overlaps must be in range (0,1)"
  @assert 0 < tol ≤ 1 "tolerance must be in range (0,1]"
  @assert cut ∈ [:dijkstra,:boykov] "invalid cut algorithm"
  @assert path ∈ [:rasterup,:rasterdown,:dilation,:random] "invalid simulation path"
  @assert nreal > 0 "invalid number of realizations"

  # soft data checks
  if !isempty(soft)
    for (aux, auxTI) in soft
      @assert ndims(aux) == 3 "soft data is not 3D (add ghost dimension for 2D)"
      @assert all([size(aux)...] .≥ [gridsizex, gridsizey, gridsizez]) "soft data size < grid size"
      @assert size(auxTI) == size(trainimg) "auxiliary TI must have the same size as TI"
    end
  end

  # hard data checks
  if !isempty(hard)
    coordinates = Int[coord[i] for coord in coords(hard), i=1:3]
    @assert all(maximum(coordinates, dims=1) .≤ [gridsizex gridsizey gridsizez]) "hard data coordinates outside of grid"
    @assert all(minimum(coordinates, dims=1) .> 0) "hard data coordinates must be positive indexes"
  end

  # calculate the overlap from given percentage
  overlapx = ceil(Int, overlapx * tilesize[1])
  overlapy = ceil(Int, overlapy * tilesize[2])
  overlapz = ceil(Int, overlapz * tilesize[3])

  # spacing in raster path
  spacingx = tilesize[1] - overlapx
  spacingy = tilesize[2] - overlapy
  spacingz = tilesize[3] - overlapz

  # calculate the number of tiles from grid size
  ntilex = ceil(Int, gridsizex / max(spacingx, 1))
  ntiley = ceil(Int, gridsizey / max(spacingy, 1))
  ntilez = ceil(Int, gridsizez / max(spacingz, 1))

  # simulation grid dimensions
  nx = ntilex * (tilesize[1] - overlapx) + overlapx
  ny = ntiley * (tilesize[2] - overlapy) + overlapy
  nz = ntilez * (tilesize[3] - overlapz) + overlapz

  # total overlap volume in simulation grid
  overlap_volume = nx*ny*nz - (nx - (ntilex-1)overlapx)*
                              (ny - (ntiley-1)overlapy)*
                              (nz - (ntilez-1)overlapz)

  # warn in case of 1-voxel overlaps
  if any((tilesize .>  1) .& ([overlapx, overlapy, overlapz] .== 1))
    warn("Overlaps with only 1 voxel. Check tilesize/overlap configuration.")
  end

  # always work with floating point
  TI = Float64.(trainimg)

  # inactive voxels in the training image
  NaNTI = isnan.(TI); TI[NaNTI] .= 0

  # perform simplex transform
  if simplex
    categories = Set(TI[.!NaNTI])
    ncategories = nvertices = length(categories) - 1

    @assert categories == Set(0:ncategories) "categories should be labeled 1, 2, 3,..."

    simplexTI = simplex_transform(TI, nvertices)
  else
    nvertices = 1
    simplexTI = [TI]
  end

  # disable tiles in the training image if they contain inactive voxels
  mₜ, nₜ, pₜ = size(TI)
  disabled = falses(mₜ-tilesize[1]+1, nₜ-tilesize[2]+1, pₜ-tilesize[3]+1)
  for nanidx in findall(vec(NaNTI))
    iₙ, jₙ, kₙ = myind2sub(size(TI), nanidx)

    # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
    iₛ = max(iₙ-tilesize[1]+1, 1)
    jₛ = max(jₙ-tilesize[2]+1, 1)
    kₛ = max(kₙ-tilesize[3]+1, 1)
    iₑ = min(iₙ, mₜ-tilesize[1]+1)
    jₑ = min(jₙ, nₜ-tilesize[2]+1)
    kₑ = min(kₙ, pₜ-tilesize[3]+1)

    disabled[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] .= true
  end

  # keep track of hard data and inactive voxels
  skipped = Set{Tuple{Int,Int,Int}}()
  datum   = Vector{Tuple{Int,Int,Int}}()
  if !isempty(hard)
    # hard data in grid format
    hardgrid = zeros(nx, ny, nz)
    preset = falses(nx, ny, nz)
    activated = trues(nx, ny, nz)
    for coord in coords(hard)
      if isnan(hard[coord])
        activated[coord...] = false
      else
        hardgrid[coord...] = hard[coord]
        preset[coord...] = true
      end
    end

    # deactivate voxels beyond true grid size
    activated[gridsizex+1:nx,:,:] .= false
    activated[:,gridsizey+1:ny,:] .= false
    activated[:,:,gridsizez+1:nz] .= false

    # grid must contain active voxels
    any_activated = any(activated[1:gridsizex,1:gridsizey,1:gridsizez])
    @assert any_activated "simulation grid has no active voxel"

    # determine tiles that should be skipped and tiles with data
    for k=1:ntilez, j=1:ntiley, i=1:ntilex
      # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
      iₛ = (i-1)spacingx + 1
      jₛ = (j-1)spacingy + 1
      kₛ = (k-1)spacingz + 1
      iₑ = iₛ + tilesize[1] - 1
      jₑ = jₛ + tilesize[2] - 1
      kₑ = kₛ + tilesize[3] - 1

      if all(.!activated[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
        push!(skipped, (i,j,k))
      else
        if any(preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
          push!(datum, (i,j,k))
        end
      end
    end
  end

  # preprocess soft data
  softgrid = Vector{Array{Float64,3}}()
  softTI   = Vector{Array{Float64,3}}()
  if !isempty(soft)
    for (aux, auxTI) in soft
      mx, my, mz = size(aux)
      lx = min(mx,nx); ly = min(my,ny); lz = min(mz,nz)

      AUX = padarray(aux, Pad(:symmetric, [0,0,0], [nx-lx,ny-ly,nz-lz]))
      AUX = parent(AUX)
      AUX[isnan.(AUX)] .= 0

      AUXTI = copy(auxTI)
      AUXTI[NaNTI] .= 0

      # always work with floating point
      AUX   = Float64.(AUX)
      AUXTI = Float64.(AUXTI)

      push!(softgrid, AUX)
      push!(softTI, AUXTI)
    end
  end

  # overwrite path option if data is available
  !isempty(datum) && (path = :datum)

  # select cut algorithm
  boundary_cut = cut == :dijkstra ? dijkstra_cut : boykov_kolmogorov_cut

  # main output is a vector of 3D grids
  realizations = Vector{Array{Float64,3}}()

  # for each realization we have:
  boundarycuts = Vector{Array{Float64,3}}() # boundary cut
  voxelreuse = Vector{Float64}()            # voxel reuse

  # show progress and estimated time duration
  showprogress && (progress = Progress(nreal, color=:black))

  # preallocate memory for distance calculations
  distance = Array{Float64}(undef, mₜ-tilesize[1]+1, nₜ-tilesize[2]+1, pₜ-tilesize[3]+1)

  for real=1:nreal
    # allocate memory for current simulation
    simgrid = zeros(nx, ny, nz)
    debug && (cutgrid = zeros(nx, ny, nz))

    # keep track of pasted tiles
    pasted = Set{Tuple{Int,Int,Int}}()

    # construct simulation path
    simpath = genpath((ntilex,ntiley,ntilez), path, datum)

    # loop simulation grid tile by tile
    for pathidx in simpath
      i, j, k = myind2sub((ntilex,ntiley,ntilez), pathidx)

      # skip tile if all voxels are inactive
      (i,j,k) ∈ skipped && continue

      # if not skipped, proceed and paste tile
      push!(pasted, (i,j,k))

      # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
      iₛ = (i-1)spacingx + 1
      jₛ = (j-1)spacingy + 1
      kₛ = (k-1)spacingz + 1
      iₑ = iₛ + tilesize[1] - 1
      jₑ = jₛ + tilesize[2] - 1
      kₑ = kₛ + tilesize[3] - 1

      # current simulation dataevent
      simdev = view(simgrid, iₛ:iₑ,jₛ:jₑ,kₛ:kₑ)

      # compute overlap distance
      distance[:] .= 0
      if overlapx > 1 && (i-1,j,k) ∈ pasted
        ovx = view(simdev,1:overlapx,:,:)
        xsimplex = simplex ? simplex_transform(ovx, nvertices) : [ovx]

        D = convdist(simplexTI, xsimplex)
        distance .+= view(D,1:mₜ-tilesize[1]+1,:,:)
      end
      if overlapx > 1 && (i+1,j,k) ∈ pasted
        ovx = view(simdev,spacingx+1:tilesize[1],:,:)
        xsimplex = simplex ? simplex_transform(ovx, nvertices) : [ovx]

        D = convdist(simplexTI, xsimplex)
        distance .+= view(D,spacingx+1:mₜ-overlapx+1,:,:)
      end
      if overlapy > 1 && (i,j-1,k) ∈ pasted
        ovy = view(simdev,:,1:overlapy,:)
        ysimplex = simplex ? simplex_transform(ovy, nvertices) : [ovy]

        D = convdist(simplexTI, ysimplex)
        distance .+= view(D,:,1:nₜ-tilesize[2]+1,:)
      end
      if overlapy > 1 && (i,j+1,k) ∈ pasted
        ovy = view(simdev,:,spacingy+1:tilesize[2],:)
        ysimplex = simplex ? simplex_transform(ovy, nvertices) : [ovy]

        D = convdist(simplexTI, ysimplex)
        distance .+= view(D,:,spacingy+1:nₜ-overlapy+1,:)
      end
      if overlapz > 1 && (i,j,k-1) ∈ pasted
        ovz = view(simdev,:,:,1:overlapz)
        zsimplex = simplex ? simplex_transform(ovz, nvertices) : [ovz]

        D = convdist(simplexTI, zsimplex)
        distance .+= view(D,:,:,1:pₜ-tilesize[3]+1)
      end
      if overlapz > 1 && (i,j,k+1) ∈ pasted
        ovz = view(simdev,:,:,spacingz+1:tilesize[3])
        zsimplex = simplex ? simplex_transform(ovz, nvertices) : [ovz]

        D = convdist(simplexTI, zsimplex)
        distance .+= view(D,:,:,spacingz+1:pₜ-overlapz+1)
      end

      # disable dataevents that contain inactive voxels
      distance[disabled] .= Inf

      # compute hard and soft distances
      auxdistances = Vector{Array{Float64,3}}()
      if !isempty(hard) && any(preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
        harddev = hardgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        hsimplex = simplex ? simplex_transform(harddev, nvertices) : [harddev]
        D = convdist(simplexTI, hsimplex, weights=preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])

        # disable dataevents that contain inactive voxels
        D[disabled] .= Inf

        # swap overlap and hard distances
        push!(auxdistances, distance)
        distance = D
      end
      for n=1:length(softTI)
        softdev = softgrid[n][iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        D = convdist([softTI[n]], [softdev])

        # disable dataevents that contain inactive voxels
        D[disabled] .= Inf

        push!(auxdistances, D)
      end

      # current pattern database
      patterndb = isempty(auxdistances) ? findall(vec(distance .≤ (1+tol)minimum(distance))) :
                                          relaxation(distance, auxdistances, tol)
      patternprobs = tau_model(patterndb, distance, auxdistances)

      # pick a pattern at random from the database
      idx = sample(patterndb, weights(patternprobs))
      iᵦ, jᵦ, kᵦ = myind2sub(size(distance), idx)

      # selected training image dataevent
      TIdev = view(TI,iᵦ:iᵦ+tilesize[1]-1,jᵦ:jᵦ+tilesize[2]-1,kᵦ:kᵦ+tilesize[3]-1)

      # boundary cut mask
      M = falses(size(simdev))
      if overlapx > 1 && (i-1,j,k) ∈ pasted
        A = view(simdev,1:overlapx,:,:); B = view(TIdev,1:overlapx,:,:)
        M[1:overlapx,:,:] .|= boundary_cut(A, B, :x)
      end
      if overlapx > 1 && (i+1,j,k) ∈ pasted
        A = view(simdev,spacingx+1:tilesize[1],:,:); B = view(TIdev,spacingx+1:tilesize[1],:,:)
        M[spacingx+1:tilesize[1],:,:] .|= reverse(boundary_cut(reverse(A, dims=1), reverse(B, dims=1), :x), dims=1)
      end
      if overlapy > 1 && (i,j-1,k) ∈ pasted
        A = view(simdev,:,1:overlapy,:); B = view(TIdev,:,1:overlapy,:)
        M[:,1:overlapy,:] .|= boundary_cut(A, B, :y)
      end
      if overlapy > 1 && (i,j+1,k) ∈ pasted
        A = view(simdev,:,spacingy+1:tilesize[2],:); B = view(TIdev,:,spacingy+1:tilesize[2],:)
        M[:,spacingy+1:tilesize[2],:] .|= reverse(boundary_cut(reverse(A, dims=2), reverse(B, dims=2), :y), dims=2)
      end
      if overlapz > 1 && (i,j,k-1) ∈ pasted
        A = view(simdev,:,:,1:overlapz); B = view(TIdev,:,:,1:overlapz)
        M[:,:,1:overlapz] .|= boundary_cut(A, B, :z)
      end
      if overlapz > 1 && (i,j,k+1) ∈ pasted
        A = view(simdev,:,:,spacingz+1:tilesize[3]); B = view(TIdev,:,:,spacingz+1:tilesize[3])
        M[:,:,spacingz+1:tilesize[3]] .|= reverse(boundary_cut(reverse(A, dims=3), reverse(B, dims=3), :z), dims=3)
      end

      # paste quilted pattern from training image
      simdev[.!M] = TIdev[.!M]

      # save boundary cut
      debug && (cutgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M)
    end

    # save voxel reuse
    debug && push!(voxelreuse, sum(cutgrid)/overlap_volume)

    # hard data and shape correction
    if !isempty(hard)
      simgrid[preset] = hardgrid[preset]
      simgrid[.!activated] .= NaN
      debug && (cutgrid[.!activated] .= NaN)
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
