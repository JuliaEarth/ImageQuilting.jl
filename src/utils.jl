# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

const GPU = nothing

function get_imfilter_impl(GPU)
  if GPU ≠ nothing
    imfilter_gpu
  else
    imfilter_cpu
  end
end

cart2lin(dims, ind) = LinearIndices(dims)[ind]
lin2cart(dims, ind) = CartesianIndices(dims)[ind]

function event!(buff, hard::Dict, tile::CartesianIndices)
  @inbounds for (i, coord) in enumerate(tile)
    if isnan(get(hard, coord, NaN))
      buff[i] = 0.0
    else
      buff[i] = hard[coord]
    end
  end
end

function event(hard::Dict, tile::CartesianIndices)
  buff = Array{Float64}(undef, size(tile))
  event!(buff, hard, tile, def)
  buff
end

function indicator!(buff, hard::Dict, tile::CartesianIndices)
  @inbounds for (i, coord) in enumerate(tile)
    if isnan(get(hard, coord, NaN))
      buff[i] = false
    else
      buff[i] = true
    end
  end
end

function indicator(hard::Dict, tile::CartesianIndices)
  buff = Array{Bool}(undef, size(tile))
  indicator!(buff, hard, tile)
  buff
end

function activation!(buff, hard::Dict, tile::CartesianIndices)
  @inbounds for (i, coord) in enumerate(tile)
    if coord ∈ keys(hard) && isnan(hard[coord])
      buff[i] = false
    else
      buff[i] = true
    end
  end
end

function activation(hard::Dict, tile::CartesianIndices)
  buff = Array{Bool}(undef, size(tile))
  activation!(buff, hard, tile)
  buff
end

function preprocess_images(trainimg::AbstractArray{T,N}, soft::AbstractVector,
                           geoconfig::NamedTuple) where {T,N}
  padsize = geoconfig.padsize

  TI = Float64.(trainimg)
  replace!(TI, NaN => 0.)

  SOFT = map(soft) do (aux, auxTI)
    prepend = ntuple(i->0, N)
    append  = padsize .- min.(padsize, size(aux))
    padding = Pad(:symmetric, prepend, append)

    AUX   = Float64.(padarray(aux, padding))
    AUXTI = Float64.(auxTI)
    replace!(AUX, NaN => 0.)
    replace!(AUXTI, NaN => 0.)

    AUX, AUXTI
  end

  TI, SOFT
end

function find_disabled(trainimg::AbstractArray{T,N}, geoconfig::NamedTuple) where {T,N}
  TIsize   = geoconfig.TIsize
  tilesize = geoconfig.tilesize
  distsize = geoconfig.distsize

  disabled = falses(distsize)
  for ind in findall(isnan, trainimg)
    start  = @. max(ind.I - tilesize + 1, 1)
    finish = @. min(ind.I, distsize)
    tile   = CartesianIndex(start):CartesianIndex(finish)
    disabled[tile] .= true
  end

  disabled
end

function find_skipped(hard::Dict, geoconfig::NamedTuple)
  ntiles   = geoconfig.ntiles
  tilesize = geoconfig.tilesize
  spacing  = geoconfig.spacing
  simsize  = geoconfig.simsize

  skipped = Set{Int}()
  datainds = Vector{Int}()
  for tileind in CartesianIndices(ntiles)
    # tile corners are given by start and finish
    start  = @. (tileind.I - 1)*spacing + 1
    finish = @. start + tilesize - 1
    tile   = CartesianIndex(start):CartesianIndex(finish)

    # skip tile if either
    #   1) tile is beyond true simulation size
    #   2) all values in the tile are NaN
    if any(start .> simsize) || !any(activation(hard, tile))
      push!(skipped, cart2lin(ntiles, tileind))
    elseif any(indicator(hard, tile))
      push!(datainds, cart2lin(ntiles, tileind))
    end
  end

  skipped, datainds
end

function genpath(extent::Dims{N}, kind::Symbol, datainds::AbstractVector{Int}) where {N}
  path = Vector{Int}()

  if isempty(datainds)
    if kind == :raster
      for ind in LinearIndices(extent)
        push!(path, ind)
      end
    end

    if kind == :random
      path = randperm(prod(extent))
    end

    if kind == :dilation
      nelm = prod(extent)
      pivot = rand(1:nelm)

      grid = falses(extent)
      grid[pivot] = true
      push!(path, pivot)

      while !all(grid)
        dilated = dilate(grid)
        append!(path, findall(vec(dilated .& .!grid)))
        grid = dilated
      end
    end
  else
    # data-first path
    shuffle!(datainds)

    grid = falses(extent)
    for pivot in datainds
      grid[pivot] = true
      push!(path, pivot)
    end

    while !all(grid)
      dilated = dilate(grid)
      append!(path, findall(vec(dilated .& .!grid)))
      grid = dilated
    end
  end

  path
end
