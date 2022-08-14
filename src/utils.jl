# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function fastdistance(img, kern; weights=fill(1.0, size(kern)))
  wkern = weights.*kern

  A² = imfilter_kernel(img.^2, weights)
  AB = imfilter_kernel(img, wkern)
  B² = sum(wkern .* kern)

  parent(@. abs(A² - 2AB + B²))
end

cart2lin(dims, ind) = LinearIndices(dims)[ind]
lin2cart(dims, ind) = CartesianIndices(dims)[ind]

function event!(buff, hard, tile)
  @inbounds for (i, coord) in enumerate(tile)
    h = get(hard, coord, NaN)
    buff[i] = ifelse(isnan(h), 0.0, h)
  end
end

function event(hard, tile)
  buff = Array{Float64}(undef, size(tile))
  event!(buff, hard, tile, def)
  buff
end

function indicator!(buff, hard, tile)
  @inbounds for (i, coord) in enumerate(tile)
    nan = isnan(get(hard, coord, NaN))
    buff[i] = ifelse(nan, false, true)
  end
end

function indicator(hard, tile)
  buff = Array{Bool}(undef, size(tile))
  indicator!(buff, hard, tile)
  buff
end

function activation!(buff, hard, tile)
  @inbounds for (i, coord) in enumerate(tile)
    cond = coord ∈ keys(hard) && isnan(hard[coord])
    buff[i] = ifelse(cond, false, true)
  end
end

function activation(hard, tile)
  buff = Array{Bool}(undef, size(tile))
  activation!(buff, hard, tile)
  buff
end

array_cpu(array) = array

array_gpu(array) = CuArray(array)

const array_kernel = CUDA.functional() ? array_gpu : array_cpu

view_cpu(array, I) = view(array, I)

view_gpu(array, I) = Array(array[I])

const view_kernel = CUDA.functional() ? view_gpu : view_cpu

function imagepreproc(trainimg, soft, geoconfig)
  padsize = geoconfig.padsize

  TI = Float64.(trainimg)
  replace!(TI, NaN => 0.)
  TI_kernel = array_kernel(TI)

  SOFT = map(soft) do (aux, auxTI)
    prepend = ntuple(i->0, ndims(TI))
    append  = padsize .- min.(padsize, size(aux))
    padding = Pad(:symmetric, prepend, append)

    AUX   = Float64.(padarray(aux, padding))
    AUXTI = Float64.(auxTI)
    replace!(AUX, NaN => 0.)
    replace!(AUXTI, NaN => 0.)

    AUX, array_kernel(AUXTI)
  end

  TI_kernel, SOFT
end

function finddisabled(trainimg, geoconfig)
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

function findskipped(hard, geoconfig)
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

function genpath(rng, extent, kind, datainds)
  path = Vector{Int}()

  if isempty(datainds)
    if kind == :raster
      for ind in LinearIndices(extent)
        push!(path, ind)
      end
    end

    if kind == :random
      path = randperm(rng, prod(extent))
    end

    if kind == :dilation
      nelm = prod(extent)
      pivot = rand(rng, 1:nelm)

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
    shuffle!(rng, datainds)

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
