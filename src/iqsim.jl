# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    iqsim(trainimg::AbstractArray{T,N},
          tilesize::NTuple{N,Int}, gridsize::NTuple{N,Int};
          overlap::NTuple{N,Float64}=ntuple(i->1/6,N),
          soft::AbstractVector=[], hard::HardData=HardData(), tol::Real=.1,
          cut::Symbol=:boykov, path::Symbol=:rasterup, nreal::Integer=1,
          threads::Integer=CPU_PHYSICAL_CORES, gpu::Bool=false,
          debug::Bool=false, showprogress::Bool=false)

Performs image quilting simulation as described in Hoffimann et al. 2017.

## Parameters

### Required

* `trainimg` is any 3D array (add ghost dimension for 2D)
* `tilesize` is the tile size (or pattern size)
* `gridsize` is the size of the simulation grid

### Optional

* `overlap` is the percentage of overlap
* `soft` is a vector of `(data,dataTI)` pairs
* `hard` is an instance of `HardData`
* `tol` is the initial relaxation tolerance in (0,1] (default to .1)
* `cut` is the cut algorithm (`:dijkstra` or `:boykov`)
* `path` is the simulation path (`:rasterup`, `:rasterdown`, `:dilation` or `:random`)
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
function iqsim(trainimg::AbstractArray{T,N},
               tilesize::NTuple{N,Int}, gridsize::NTuple{N,Int};
               overlap::NTuple{N,Float64}=ntuple(i->1/6,N),
               soft::AbstractVector=[], hard::HardData=HardData(), tol::Real=.1,
               cut::Symbol=:boykov, path::Symbol=:rasterup, nreal::Integer=1,
               threads::Integer=CPU_PHYSICAL_CORES, gpu::Bool=false,
               debug::Bool=false, showprogress::Bool=false) where {T,N}

  # number of threads in FFTW
  set_num_threads(threads)

  # sanity checks
  @assert ndims(trainimg) == 3 "image is not 3D (add ghost dimension for 2D)"
  @assert all(0 .< tilesize .≤ size(trainimg)) "invalid tile size"
  @assert all(gridsize .≥ tilesize) "invalid grid size"
  @assert all(0 .< overlap .< 1) "overlaps must be in range (0,1)"
  @assert 0 < tol ≤ 1 "tolerance must be in range (0,1]"
  @assert cut ∈ [:dijkstra,:boykov] "invalid cut algorithm"
  @assert path ∈ [:rasterup,:rasterdown,:dilation,:random] "invalid simulation path"
  @assert nreal > 0 "invalid number of realizations"

  # soft data checks
  if !isempty(soft)
    for (aux, auxTI) in soft
      @assert ndims(aux) == 3 "soft data is not 3D (add ghost dimension for 2D)"
      @assert all(size(aux) .≥ gridsize) "soft data size < grid size"
      @assert size(auxTI) == size(trainimg) "auxiliary TI must have the same size as TI"
    end
  end

  # hard data checks
  if !isempty(hard)
    coordinates = Int[coord[i] for coord in coords(hard), i=1:3]
    @assert all(maximum(coordinates, dims=1)' .≤ gridsize) "hard data coordinates outside of grid"
    @assert all(minimum(coordinates, dims=1)' .> 0) "hard data coordinates must be positive indexes"
  end

  # calculate the overlap size from given percentage
  ovlsize = ntuple(i -> ceil(Int, overlap[i]*tilesize[i]), N)

  # spacing in raster path
  spacing = ntuple(i -> tilesize[i] - ovlsize[i], N)

  # calculate the number of tiles from grid size
  ntiles = ntuple(i -> ceil(Int, gridsize[i]/max(spacing[i],1)), N)

  # simulation grid dimensions
  padsize = ntuple(i -> ntiles[i]*(tilesize[i]-ovlsize[i]) + ovlsize[i], N)

  # total overlap volume in simulation grid
  ovlvol = prod(padsize) - prod(padsize[i] - (ntiles[i]-1)*ovlsize[i] for i in 1:N)

  # warn in case of 1-voxel overlaps
  if any((tilesize .>  1) .& (ovlsize .== 1))
    warn("Overlaps with only 1 voxel. Check tilesize/overlap configuration.")
  end

  # always work with floating point
  TI = Float64.(trainimg)

  # inactive voxels in the training image
  NaNTI = isnan.(TI); TI[NaNTI] .= 0

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
    hardgrid = zeros(padsize)
    preset = falses(padsize)
    activated = trues(padsize)
    for coord in coords(hard)
      if isnan(hard[coord])
        activated[coord...] = false
      else
        hardgrid[coord...] = hard[coord]
        preset[coord...] = true
      end
    end

    # deactivate voxels beyond true grid size
    activated[gridsize[1]+1:padsize[1],:,:] .= false
    activated[:,gridsize[2]+1:padsize[2],:] .= false
    activated[:,:,gridsize[3]+1:padsize[3]] .= false

    # grid must contain active voxels
    any_activated = any(activated[1:gridsize[1],1:gridsize[2],1:gridsize[3]])
    @assert any_activated "simulation grid has no active voxel"

    # determine tiles that should be skipped and tiles with data
    for k=1:ntiles[3], j=1:ntiles[2], i=1:ntiles[1]
      # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
      iₛ = (i-1)spacing[1] + 1
      jₛ = (j-1)spacing[2] + 1
      kₛ = (k-1)spacing[3] + 1
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
      auxpad = padsize .- min.(padsize, size(aux))

      AUX = padarray(aux, Pad(:symmetric, ntuple(i->0,N), auxpad))
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
    simgrid = zeros(padsize)
    debug && (cutgrid = zeros(padsize))

    # keep track of pasted tiles
    pasted = Set{NTuple{N,Int}}()

    # construct simulation path
    simpath = genpath(ntiles, path, datum)

    # loop simulation grid tile by tile
    for pathidx in simpath
      i, j, k = myind2sub(ntiles, pathidx)

      # skip tile if all voxels are inactive
      (i,j,k) ∈ skipped && continue

      # if not skipped, proceed and paste tile
      push!(pasted, (i,j,k))

      # tile corners are given by (iₛ,jₛ,kₛ) and (iₑ,jₑ,kₑ)
      iₛ = (i-1)spacing[1] + 1
      jₛ = (j-1)spacing[2] + 1
      kₛ = (k-1)spacing[3] + 1
      iₑ = iₛ + tilesize[1] - 1
      jₑ = jₛ + tilesize[2] - 1
      kₑ = kₛ + tilesize[3] - 1

      # current simulation dataevent
      simdev = view(simgrid, iₛ:iₑ,jₛ:jₑ,kₛ:kₑ)

      # compute overlap distance
      distance[:] .= 0
      if ovlsize[1] > 1 && (i-1,j,k) ∈ pasted
        ovx = view(simdev,1:ovlsize[1],:,:)
        xsimplex = [ovx]

        D = convdist([TI], xsimplex)
        distance .+= view(D,1:mₜ-tilesize[1]+1,:,:)
      end
      if ovlsize[1] > 1 && (i+1,j,k) ∈ pasted
        ovx = view(simdev,spacing[1]+1:tilesize[1],:,:)
        xsimplex = [ovx]

        D = convdist([TI], xsimplex)
        distance .+= view(D,spacing[1]+1:mₜ-ovlsize[1]+1,:,:)
      end
      if ovlsize[2] > 1 && (i,j-1,k) ∈ pasted
        ovy = view(simdev,:,1:ovlsize[2],:)
        ysimplex = [ovy]

        D = convdist([TI], ysimplex)
        distance .+= view(D,:,1:nₜ-tilesize[2]+1,:)
      end
      if ovlsize[2] > 1 && (i,j+1,k) ∈ pasted
        ovy = view(simdev,:,spacing[2]+1:tilesize[2],:)
        ysimplex = [ovy]

        D = convdist([TI], ysimplex)
        distance .+= view(D,:,spacing[2]+1:nₜ-ovlsize[2]+1,:)
      end
      if ovlsize[3] > 1 && (i,j,k-1) ∈ pasted
        ovz = view(simdev,:,:,1:ovlsize[3])
        zsimplex = [ovz]

        D = convdist([TI], zsimplex)
        distance .+= view(D,:,:,1:pₜ-tilesize[3]+1)
      end
      if ovlsize[3] > 1 && (i,j,k+1) ∈ pasted
        ovz = view(simdev,:,:,spacing[3]+1:tilesize[3])
        zsimplex = [ovz]

        D = convdist([TI], zsimplex)
        distance .+= view(D,:,:,spacing[3]+1:pₜ-ovlsize[3]+1)
      end

      # disable dataevents that contain inactive voxels
      distance[disabled] .= Inf

      # compute hard and soft distances
      auxdistances = Vector{Array{Float64,3}}()
      if !isempty(hard) && any(preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])
        harddev = hardgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ]
        hsimplex = [harddev]
        D = convdist([TI], hsimplex, weights=preset[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ])

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
      if ovlsize[1] > 1 && (i-1,j,k) ∈ pasted
        A = view(simdev,1:ovlsize[1],:,:); B = view(TIdev,1:ovlsize[1],:,:)
        M[1:ovlsize[1],:,:] .|= boundary_cut(A, B, :x)
      end
      if ovlsize[1] > 1 && (i+1,j,k) ∈ pasted
        A = view(simdev,spacing[1]+1:tilesize[1],:,:); B = view(TIdev,spacing[1]+1:tilesize[1],:,:)
        M[spacing[1]+1:tilesize[1],:,:] .|= reverse(boundary_cut(reverse(A, dims=1), reverse(B, dims=1), :x), dims=1)
      end
      if ovlsize[2] > 1 && (i,j-1,k) ∈ pasted
        A = view(simdev,:,1:ovlsize[2],:); B = view(TIdev,:,1:ovlsize[2],:)
        M[:,1:ovlsize[2],:] .|= boundary_cut(A, B, :y)
      end
      if ovlsize[2] > 1 && (i,j+1,k) ∈ pasted
        A = view(simdev,:,spacing[2]+1:tilesize[2],:); B = view(TIdev,:,spacing[2]+1:tilesize[2],:)
        M[:,spacing[2]+1:tilesize[2],:] .|= reverse(boundary_cut(reverse(A, dims=2), reverse(B, dims=2), :y), dims=2)
      end
      if ovlsize[3] > 1 && (i,j,k-1) ∈ pasted
        A = view(simdev,:,:,1:ovlsize[3]); B = view(TIdev,:,:,1:ovlsize[3])
        M[:,:,1:ovlsize[3]] .|= boundary_cut(A, B, :z)
      end
      if ovlsize[3] > 1 && (i,j,k+1) ∈ pasted
        A = view(simdev,:,:,spacing[3]+1:tilesize[3]); B = view(TIdev,:,:,spacing[3]+1:tilesize[3])
        M[:,:,spacing[3]+1:tilesize[3]] .|= reverse(boundary_cut(reverse(A, dims=3), reverse(B, dims=3), :z), dims=3)
      end

      # paste quilted pattern from training image
      simdev[.!M] = TIdev[.!M]

      # save boundary cut
      debug && (cutgrid[iₛ:iₑ,jₛ:jₑ,kₛ:kₑ] = M)
    end

    # save voxel reuse
    debug && push!(voxelreuse, sum(cutgrid)/ovlvol)

    # hard data and shape correction
    if !isempty(hard)
      simgrid[preset] = hardgrid[preset]
      simgrid[.!activated] .= NaN
      debug && (cutgrid[.!activated] .= NaN)
    end

    # throw away voxels that are outside of the grid
    simgrid = view(simgrid,1:gridsize[1],1:gridsize[2],1:gridsize[3])
    debug && (cutgrid = view(cutgrid,1:gridsize[1],1:gridsize[2],1:gridsize[3]))

    # save and continue
    push!(realizations, simgrid)
    debug && push!(boundarycuts, cutgrid)

    # update progress bar
    showprogress && next!(progress)
  end

  debug ? (realizations, boundarycuts, voxelreuse) : realizations
end
