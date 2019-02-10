# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    iqsim(trainimg::AbstractArray{T,N}, tilesize::Dims{N},
          simsize::Dims{N}=size(trainimg);
          overlap::NTuple{N,Float64}=ntuple(i->1/6,N),
          soft::AbstractVector=[], hard::Dict=Dict(), tol::Real=.1,
          cut::Symbol=:boykov, path::Symbol=:raster, nreal::Integer=1,
          threads::Integer=cpucores(), gpu::Bool=false,
          debug::Bool=false, showprogress::Bool=false)

Performs image quilting simulation as described in Hoffimann et al. 2017.

## Parameters

### Required

* `trainimg` is any 3D array (add ghost dimension for 2D)
* `tilesize` is the tile size (or pattern size)

### Optional

* `simsize` is the size of the simulation grid (default to training image size)
* `overlap` is the percentage of overlap (default to 1/6 of tile size)
* `soft` is a vector of `(data,dataTI)` pairs (default to none)
* `hard` is a dictionary mapping coordinates to data values (default to none)
* `tol` is the initial relaxation tolerance in (0,1] (default to .1)
* `cut` is the cut algorithm (`:dijkstra` or `:boykov`)
* `path` is the simulation path (`:raster`, `:dilation` or `:random`)
* `nreal` is the number of realizations (default to 1)
* `threads` is the number of threads for the FFT (default to all CPU cores)
* `gpu` informs whether to use the GPU or the CPU (default to false)
* `debug` informs whether to export or not the boundary cuts and voxel reuse
* `showprogress` informs whether to show or not estimated time duration

The main output `reals` consists of a list of 3D realizations that can be indexed with
`reals[1], reals[2], ..., reals[nreal]`. If `debug=true`, additional output is generated:

```julia
reals, cuts, voxs = iqsim(..., debug=true)
```

`cuts[i]` is the boundary cut for `reals[i]` and `voxs[i]` is the associated voxel reuse.
"""
function iqsim(trainimg::AbstractArray{T,N}, tilesize::Dims{N},
               simsize::Dims{N}=size(trainimg);
               overlap::NTuple{N,Float64}=ntuple(i->1/6,N),
               soft::AbstractVector=[], hard::Dict=Dict(), tol::Real=.1,
               cut::Symbol=:boykov, path::Symbol=:raster, nreal::Integer=1,
               threads::Integer=cpucores(), gpu::Bool=false,
               debug::Bool=false, showprogress::Bool=false) where {T,N}

  # number of threads in FFTW
  set_num_threads(threads)

  # sanity checks
  @assert ndims(trainimg) == 3 "image is not 3D (add ghost dimension for 2D)"
  @assert all(0 .< tilesize .≤ size(trainimg)) "invalid tile size"
  @assert all(simsize .≥ tilesize) "invalid grid size"
  @assert all(0 .< overlap .< 1) "overlaps must be in range (0,1)"
  @assert 0 < tol ≤ 1 "tolerance must be in range (0,1]"
  @assert cut ∈ [:dijkstra,:boykov] "invalid cut algorithm"
  @assert path ∈ [:raster,:dilation,:random] "invalid simulation path"
  @assert nreal > 0 "invalid number of realizations"

  # soft data checks
  if !isempty(soft)
    for (aux, auxTI) in soft
      @assert ndims(aux) == 3 "soft data is not 3D (add ghost dimension for 2D)"
      @assert all(size(aux) .≥ simsize) "soft data size < grid size"
      @assert size(auxTI) == size(trainimg) "auxiliary TI must have the same size as TI"
    end
  end

  # hard data checks
  if !isempty(hard)
    coords = [coord[i] for i in 1:N, coord in keys(hard)]
    @assert all(maximum(coords, dims=2) .≤ simsize) "hard data coordinates outside of grid"
    @assert all(minimum(coords, dims=2) .> 0) "hard data coordinates must be positive indices"
  end

  # calculate the overlap size from given percentage
  ovlsize = @. ceil(Int, overlap*tilesize)

  # spacing in raster path
  spacing = @. tilesize - ovlsize

  # calculate the number of tiles from grid size
  ntiles = @. ceil(Int, simsize / max(spacing, 1))

  # simulation grid dimensions
  padsize = @. ntiles*(tilesize - ovlsize) + ovlsize

  # training image dimensions
  TIsize = size(trainimg)

  # total overlap volume in simulation grid
  ovlvol = prod(padsize) - prod(@. padsize - (ntiles - 1)*ovlsize)

  # warn in case of 1-voxel overlaps
  if any((tilesize .>  1) .& (ovlsize .== 1))
    @warn "Overlaps with only 1 voxel, check tilesize/overlap configuration"
  end

  # pad input images and knockout inactive voxels
  TI, SOFT = preprocess_images(trainimg, soft, padsize)

  # disable tiles in the training image if they contain inactive voxels
  disabled = find_disabled(trainimg, tilesize)

  # keep track of hard data and inactive voxels
  skipped  = Set{Int}()
  datainds = Vector{Int}()
  if !isempty(hard)
    # hard data in grid format
    hardgrid = zeros(padsize)
    preset = falses(padsize)
    activated = trues(padsize)
    for coord in keys(hard)
      if isnan(hard[coord])
        activated[coord] = false
      else
        hardgrid[coord] = hard[coord]
        preset[coord] = true
      end
    end

    # deactivate voxels beyond true grid size
    ax = axes(activated)
    for d=1:N
      slice = ntuple(i -> i == d ? (simsize[d]+1:padsize[d]) : ax[i], N)
      activated[CartesianIndices(slice)] .= false
    end

    # grid must contain active voxels
    @assert any(activated[CartesianIndices(simsize)]) "simulation grid has no active voxel"

    # determine tiles that should be skipped and tiles with data
    for tileind in CartesianIndices(ntiles)
      # tile corners are given by start and finish
      start  = @. (tileind.I - 1)*spacing + 1
      finish = @. start + tilesize - 1
      tile   = CartesianIndex(start):CartesianIndex(finish)

      if all(.!activated[tile])
        push!(skipped, cart2lin(ntiles, tileind))
      else
        if any(preset[tile])
          push!(datainds, cart2lin(ntiles, tileind))
        end
      end
    end
  end

  # construct simulation path
  simpath = genpath(ntiles, path, datainds)

  # select cut algorithm
  boundary_cut = cut == :dijkstra ? dijkstra_cut : boykov_kolmogorov_cut

  # show progress and estimated time duration
  showprogress && (progress = Progress(nreal))

  # main output is a vector of 3D grids
  realizations = Vector{Array{Float64,N}}()

  # for each realization we have:
  boundarycuts = Vector{Array{Float64,N}}()
  voxelreuse   = Vector{Float64}()

  # preallocate memory for distance calculations
  distance = Array{Float64}(undef, TIsize .- tilesize .+ 1)

  # preallocate memory for cut mask
  mask = Array{Bool}(undef, tilesize)

  for real=1:nreal
    # allocate memory for current simulation
    simgrid = zeros(padsize)
    debug && (cutgrid = zeros(padsize))

    # keep track of pasted tiles
    pasted = Set{CartesianIndex{N}}()

    # loop simulation grid tile by tile
    for ind in simpath
      # skip tile if all voxels are inactive
      ind ∈ skipped && continue

      # Cartesian index of tile
      tileind = lin2cart(ntiles, ind)

      # if not skipped, proceed and paste tile
      push!(pasted, tileind)

      # tile corners are given by start and finish
      start  = @. (tileind.I - 1)*spacing + 1
      finish = @. start + tilesize - 1
      tile   = CartesianIndex(start):CartesianIndex(finish)

      # current simulation dataevent
      simdev = view(simgrid, tile)

      # compute overlap distance
      distance .= 0
      for d=1:N
        # Cartesian index of previous and next tiles along dimension
        prev = CartesianIndex(ntuple(i -> i == d ? (tileind[d]-1) : tileind[i], N))
        next = CartesianIndex(ntuple(i -> i == d ? (tileind[d]+1) : tileind[i], N))

        # compute overlap distance with previous tile
        if ovlsize[d] > 1 && prev ∈ pasted
          oslice = ntuple(i -> i == d ? (1:ovlsize[d]) : (1:tilesize[i]), N)
          ovl = view(simdev, CartesianIndices(oslice))

          D = convdist(TI, ovl)

          ax = axes(D)
          dslice = ntuple(i -> i == d ? (1:TIsize[d]-tilesize[d]+1) : ax[i], N)
          distance .+= view(D, CartesianIndices(dslice))
        end

        # compute overlap distance with next tile
        if ovlsize[d] > 1 && next ∈ pasted
          oslice = ntuple(i -> i == d ? (spacing[d]+1:tilesize[d]) : (1:tilesize[i]), N)
          ovl = view(simdev, CartesianIndices(oslice))

          D = convdist(TI, ovl)

          ax = axes(D)
          dslice = ntuple(i -> i == d ? (spacing[d]+1:TIsize[d]-ovlsize[d]+1) : ax[i], N)
          distance .+= view(D, CartesianIndices(dslice))
        end
      end

      # disable dataevents that contain inactive voxels
      distance[disabled] .= Inf

      # compute hard and soft distances
      auxdistances = Vector{Array{Float64,N}}()
      if !isempty(hard) && any(preset[tile])
        harddev = view(hardgrid, tile)
        D = convdist(TI, harddev, weights=preset[tile])

        # disable dataevents that contain inactive voxels
        D[disabled] .= Inf

        # swap overlap and hard distances
        push!(auxdistances, distance)
        distance = D
      end
      for (AUX, AUXTI) in SOFT
        softdev = view(AUX, tile)
        D = convdist(AUXTI, softdev)

        # disable dataevents that contain inactive voxels
        D[disabled] .= Inf

        push!(auxdistances, D)
      end

      # current pattern database
      patterndb = isempty(auxdistances) ? findall(vec(distance .≤ (1+tol)minimum(distance))) :
                                          relaxation(distance, auxdistances, tol)

      # pattern probability
      patternprobs = tau_model(patterndb, distance, auxdistances)

      # pick a pattern at random from the database
      rind = sample(patterndb, weights(patternprobs))
      start  = lin2cart(size(distance), rind)
      finish = @. start.I + tilesize - 1
      rtile  = CartesianIndex(start):CartesianIndex(finish)

      # selected training image dataevent
      TIdev = view(TI, rtile)

      # boundary cut mask
      mask .= false
      for d=1:N
        # Cartesian index of previous and next tiles along dimension
        prev = CartesianIndex(ntuple(i -> i == d ? (tileind[d]-1) : tileind[i], N))
        next = CartesianIndex(ntuple(i -> i == d ? (tileind[d]+1) : tileind[i], N))

        # compute mask with previous tile
        if ovlsize[d] > 1 && prev ∈ pasted
          oslice = ntuple(i -> i == d ? (1:ovlsize[d]) : (1:tilesize[i]), N)
          inds = CartesianIndices(oslice)
          A = view(simdev, inds); B = view(TIdev, inds)
          mask[inds] .|= boundary_cut(A, B, d)
        end

        # compute mask with next tile
        if ovlsize[d] > 1 && next ∈ pasted
          oslice = ntuple(i -> i == d ? (spacing[d]+1:tilesize[d]) : (1:tilesize[i]), N)
          inds = CartesianIndices(oslice)
          A = view(simdev, inds); B = view(TIdev, inds)
          mask[inds] .|= reverse(boundary_cut(reverse(A, dims=d), reverse(B, dims=d), d), dims=d)
        end
      end

      # paste quilted pattern from training image
      simdev[.!mask] = TIdev[.!mask]

      # save boundary cut
      debug && (cutgrid[tile] = mask)
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
    simgrid = view(simgrid, CartesianIndices(simsize))
    debug && (cutgrid = view(cutgrid, CartesianIndices(simsize)))

    # save and continue
    push!(realizations, simgrid)
    debug && push!(boundarycuts, cutgrid)

    # update progress bar
    showprogress && next!(progress)
  end

  debug ? (realizations, boundarycuts, voxelreuse) : realizations
end

function preprocess_images(trainimg::AbstractArray{T,N}, soft::AbstractVector,
                           padsize::Dims{N}) where {T,N}
  # training image
  TI = copy(trainimg)
  TI[isnan.(TI)] .= 0

  # soft data
  SOFT = map(soft) do (aux, auxTI)
    auxpad = padsize .- min.(padsize, size(aux))

    AUX = padarray(aux, Pad(:symmetric, ntuple(i->0, N), auxpad))
    AUX = parent(AUX)

    AUXTI = copy(auxTI)

    AUX[isnan.(AUX)] .= 0
    AUXTI[isnan.(AUXTI)] .= 0

    AUX, AUXTI
  end

  TI, SOFT
end

function find_disabled(trainimg::AbstractArray{T,N}, tilesize::Dims{N}) where {T,N}
  TIsize = size(trainimg)
  disabled = falses(TIsize .- tilesize .+ 1)
  for ind in findall(isnan, trainimg)
    start  = @. max(ind.I - tilesize + 1, 1)
    finish = @. min(ind.I, TIsize - tilesize + 1)
    tile   = CartesianIndex(start):CartesianIndex(finish)
    disabled[tile] .= true
  end

  disabled
end
