# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

const CPU_PHYSICAL_CORES = cpucores()

function get_imfilter_impl(GPU)
  if GPU ≠ nothing
    imfilter_gpu
  else
    imfilter_cpu
  end
end

function convdist(Xs::AbstractArray, masks::AbstractArray; weights=nothing)
  # choose among imfilter implementations
  imfilter_impl = get_imfilter_impl(GPU)

  # default to uniform weights
  weights == nothing && (weights = ones(masks[1]))

  result = []
  for (X, mask) in zip(Xs, masks)
    wmask = weights.*mask

    A² = imfilter_impl(X.^2, weights)
    AB = imfilter_impl(X, wmask)
    B² = sum(abs2, wmask)

    push!(result, abs.(A² - 2AB + B²))
  end

  D = sum(result)

  # always return a plain simple array
  parent(D)
end

function genpath(extent::NTuple{3,Integer}, kind::Symbol, datum=[])
  path = Int[]

  if kind == :rasterup
    for k=1:extent[3], j=1:extent[2], i=1:extent[1]
      push!(path, sub2ind(extent, i,j,k))
    end
  end

  if kind == :rasterdown
    for k=extent[3]:-1:1, j=1:extent[2], i=1:extent[1]
      push!(path, sub2ind(extent, i,j,k))
    end
  end

  if kind == :random
    nelm = prod(extent)
    path = nthperm!(collect(1:nelm), rand(1:factorial(big(nelm))))
  end

  if kind == :dilation
    nelm = prod(extent)
    pivot = rand(1:nelm)

    grid = falses(extent)
    grid[pivot] = true
    push!(path, pivot)

    while !all(grid)
      dilated = dilate(grid, [1,2,3])
      append!(path, find(dilated - grid))
      grid = dilated
    end
  end

  if kind == :datum
    @assert !isempty(datum) "datum path cannot be generated without data"

    shuffle!(datum)

    grid = falses(extent)
    for (i,j,k) in datum
      pivot = sub2ind(extent, i,j,k)
      grid[pivot] = true
      push!(path, pivot)
    end

    while !all(grid)
      dilated = dilate(grid, [1,2,3])
      append!(path, find(dilated - grid))
      grid = dilated
    end
  end

  path
end
