# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    iqsim(trainimg::AbstractArray{T,N}, tilesize::Dims{N},
          simsize::Dims{N}=size(trainimg);
          overlap::NTuple{N,<:Real}=ntuple(i->1/6,N),
          soft::AbstractVector=[], hard::Dict=Dict(), tol::Real=.1,
          path::Symbol=:raster, nreal::Integer=1,
          debug::Bool=false, showprogress::Bool=true,
          rng::AbstractRNG=Random.default_rng())

Performs image quilting simulation as described in Hoffimann et al. 2017.

## Parameters

### Required

* `trainimg` is any Julia array
* `tilesize` is the tile size

### Optional

* `simsize` is the size of the simulation grid (default to training image size)
* `overlap` is the percentage of overlap (default to 1/6 of tile size)
* `soft` is a vector of `(data,dataTI)` pairs (default to none)
* `hard` is a dictionary mapping coordinates to data values (default to none)
* `tol` is the initial relaxation tolerance in (0,1] (default to .1)
* `path` is the simulation path (`:raster`, `:dilation` or `:random`)
* `nreal` is the number of realizations (default to 1)
* `debug` informs whether to export or not the boundary cuts and voxel reuse
* `showprogress` informs whether to show or not estimated time duration
* `rng` is the random number generator (default to `Random.default_rng()`)

The main output `reals` consists of a list of realizations that can be indexed with
`reals[1], reals[2], ..., reals[nreal]`. If `debug=true`, additional output is generated:

```julia
reals, cuts, voxs = iqsim(..., debug=true)
```

`cuts[i]` is the boundary cut for `reals[i]` and `voxs[i]` is the associated voxel reuse.
"""
function iqsim(
  trainimg::AbstractArray{T,N},
  tilesize::Dims{N},
  simsize::Dims{N}=size(trainimg);
  overlap::NTuple{N,<:Real}=ntuple(i -> 1 / 6, N),
  soft::AbstractVector=[],
  hard::Dict=Dict(),
  tol::Real=0.1,
  path::Symbol=:raster,
  nreal::Integer=1,
  debug::Bool=false,
  showprogress::Bool=true,
  rng=Random.default_rng()
) where {T,N}

  # number of threads in FFTW
  set_num_threads(cpucores())

  # sanity checks
  @assert all(0 .< tilesize .≤ size(trainimg)) "invalid tile size"
  @assert all(simsize .≥ tilesize) "invalid grid size"
  @assert all(0 .< overlap .< 1) "overlaps must be in range (0,1)"
  @assert 0 < tol ≤ 1 "tolerance must be in range (0,1]"
  @assert path ∈ [:raster, :dilation, :random] "invalid simulation path"
  @assert nreal > 0 "invalid number of realizations"

  # soft data checks
  if !isempty(soft)
    for (aux, auxTI) in soft
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
  ovlsize = @. ceil(Int, overlap * tilesize)

  # warn in case of 1-voxel overlaps
  if any((tilesize .> 1) .& (ovlsize .== 1))
    @warn "Overlaps with only 1 voxel, check tilesize/overlap configuration"
  end

  # spacing in raster path
  spacing = @. tilesize - ovlsize

  # calculate the number of tiles from grid size
  ntiles = @. ceil(Int, simsize / max(spacing, 1))

  # padded simulation grid size
  padsize = @. ntiles * (tilesize - ovlsize) + ovlsize

  # training image dimensions
  TIsize = size(trainimg)

  # distance matrix size
  distsize = TIsize .- tilesize .+ 1

  # total overlap volume in simulation grid
  ovlvol = prod(padsize) - prod(@. padsize - (ntiles - 1) * ovlsize)

  # geometric configuration
  geoconfig = (
    ntiles=ntiles,
    tilesize=tilesize,
    ovlsize=ovlsize,
    spacing=spacing,
    TIsize=TIsize,
    simsize=simsize,
    padsize=padsize,
    distsize=distsize
  )

  # pad input images and knockout inactive voxels
  TI, SOFT = imagepreproc(trainimg, soft, geoconfig)

  # disable tiles in the training image if they contain inactive voxels
  disabled = finddisabled(trainimg, geoconfig)

  # determine tiles that should be skipped and tiles with data
  skipped, datainds = findskipped(hard, geoconfig)

  # construct simulation path
  simpath = genpath(rng, ntiles, path, datainds)

  # show progress and estimated time duration
  showprogress && (progress = Progress(nreal))

  # main output is a vector of grids
  realizations = Vector{Array{Float64,N}}()

  # for each realization we have:
  boundarycuts = Vector{Array{Float64,N}}()
  voxelreuse = Vector{Float64}()

  # preallocate memory
  ovlmask = BitArray(undef, tilesize)
  cutmask = BitArray(undef, tilesize)
  ovldist = Array{Float64}(undef, distsize)
  softdists = [Array{Float64}(undef, distsize) for i in 1:length(soft)]
  if !isempty(hard)
    hardmask = BitArray(undef, tilesize)
    harddev = Array{Float64}(undef, tilesize)
    harddist = Array{Float64}(undef, distsize)
  end

  for real in 1:nreal
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

      # tile corners are given by start and finish
      start = @. (tileind.I - 1) * spacing + 1
      finish = @. start + tilesize - 1
      tile = CartesianIndex(start):CartesianIndex(finish)

      # current simulation dataevent
      simdev = view(simgrid, tile)

      # overlap distance
      ovlmask .= false
      for d in 1:N
        # Cartesian index of previous and next tiles along dimension
        prev = CartesianIndex(ntuple(i -> i == d ? (tileind[d] - 1) : tileind[i], N))
        next = CartesianIndex(ntuple(i -> i == d ? (tileind[d] + 1) : tileind[i], N))

        # overlap with previous tile
        if ovlsize[d] > 1 && prev ∈ pasted
          oslice = ntuple(i -> i == d ? (1:ovlsize[d]) : (1:tilesize[i]), N)
          ovlmask[CartesianIndices(oslice)] .= true
        end

        # overlap with next tile
        if ovlsize[d] > 1 && next ∈ pasted
          oslice = ntuple(i -> i == d ? ((spacing[d] + 1):tilesize[d]) : (1:tilesize[i]), N)
          ovlmask[CartesianIndices(oslice)] .= true
        end
      end
      ovldist .= fastdistance(TI, simdev, weights=ovlmask)
      ovldist[disabled] .= Inf

      # hard distance
      hardtile = false
      if !isempty(hard)
        indicator!(hardmask, hard, tile)
        if any(hardmask)
          event!(harddev, hard, tile)
          harddist .= fastdistance(TI, harddev, weights=hardmask)
          harddist[disabled] .= Inf
          hardtile = true
        end
      end

      # soft distance
      for s in eachindex(SOFT)
        AUX, AUXTI = SOFT[s]
        softdev = view(AUX, tile)
        softdists[s] .= fastdistance(AUXTI, softdev)
        softdists[s][disabled] .= Inf
      end

      # main and auxiliary distances
      if hardtile
        D, Ds = harddist, [ovldist, softdists...]
      else
        D, Ds = ovldist, softdists
      end

      # current pattern database
      patterndb = isempty(Ds) ? findall(vec(D .≤ (1 + tol)minimum(D))) : relaxation(D, Ds, tol)

      # pattern probability
      patternprobs = taumodel(patterndb, D, Ds)

      # pick a pattern at random from the database
      rind = sample(rng, patterndb, weights(patternprobs))
      start = lin2cart(distsize, rind)
      finish = @. start.I + tilesize - 1
      rtile = CartesianIndex(start):CartesianIndex(finish)

      # selected training image dataevent
      TIdev = view_kernel(TI, rtile)

      # boundary cut mask
      cutmask .= false
      for d in 1:N
        # Cartesian index of previous and next tiles along dimension
        prev = CartesianIndex(ntuple(i -> i == d ? (tileind[d] - 1) : tileind[i], N))
        next = CartesianIndex(ntuple(i -> i == d ? (tileind[d] + 1) : tileind[i], N))

        # compute mask with previous tile
        if ovlsize[d] > 1 && prev ∈ pasted
          oslice = ntuple(i -> i == d ? (1:ovlsize[d]) : (1:tilesize[i]), N)
          inds = CartesianIndices(oslice)
          A = view(simdev, inds)
          B = view(TIdev, inds)
          cutmask[inds] .|= graphcut(A, B, d)
        end

        # compute mask with next tile
        if ovlsize[d] > 1 && next ∈ pasted
          oslice = ntuple(i -> i == d ? ((spacing[d] + 1):tilesize[d]) : (1:tilesize[i]), N)
          inds = CartesianIndices(oslice)
          A = view(simdev, inds)
          B = view(TIdev, inds)
          cutmask[inds] .|= .!graphcut(A, B, d)
        end
      end

      # paste quilted pattern from training image
      simdev[.!cutmask] = TIdev[.!cutmask]

      # save boundary cut
      debug && (cutgrid[tile] = cutmask)

      # mark tile as pasted
      push!(pasted, tileind)
    end

    # save voxel reuse
    debug && push!(voxelreuse, sum(cutgrid) / ovlvol)

    # hard data and shape correction
    if !isempty(hard)
      for (coord, val) in hard
        simgrid[coord] = val
      end
      if debug
        for (coord, val) in hard
          isnan(val) && (cutgrid[coord] = val)
        end
      end
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
