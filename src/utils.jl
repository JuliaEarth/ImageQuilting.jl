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
