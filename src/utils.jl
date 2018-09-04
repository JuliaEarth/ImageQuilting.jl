# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

const CPU_PHYSICAL_CORES = cpucores()

const GPU = nothing

function get_imfilter_impl(GPU)
  if GPU ≠ nothing
    imfilter_gpu
  else
    imfilter_cpu
  end
end

mysub2ind(dims, I...) = LinearIndices(dims)[I...]
myind2sub(dims, ind)  = Tuple(CartesianIndices(dims)[ind])

function convdist(img::AbstractArray, kern::AbstractArray;
                  weights::AbstractArray=fill(1.0, size(kern)))
  # choose among imfilter implementations
  imfilter_impl = get_imfilter_impl(GPU)

  wkern = weights.*kern

  A² = imfilter_impl(img.^2, weights)
  AB = imfilter_impl(img, wkern)
  B² = sum(abs2, wkern)

  D = abs.(A² .- 2AB .+ B²)

  # always return a plain simple array
  parent(D)
end

function genpath(extent::NTuple{3,Integer}, kind::Symbol, datum=[])
  path = Vector{Int}()

  if kind == :rasterup
    for k=1:extent[3], j=1:extent[2], i=1:extent[1]
      push!(path, mysub2ind(extent, i,j,k))
    end
  end

  if kind == :rasterdown
    for k=extent[3]:-1:1, j=1:extent[2], i=1:extent[1]
      push!(path, mysub2ind(extent, i,j,k))
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
      append!(path, findall(vec(dilated .& .!grid)))
      grid = dilated
    end
  end

  if kind == :datum
    @assert !isempty(datum) "datum path cannot be generated without data"

    shuffle!(datum)

    grid = falses(extent)
    for (i,j,k) in datum
      pivot = mysub2ind(extent, i,j,k)
      grid[pivot] = true
      push!(path, pivot)
    end

    while !all(grid)
      dilated = dilate(grid, [1,2,3])
      append!(path, findall(vec(dilated .& .!grid)))
      grid = dilated
    end
  end

  path
end
