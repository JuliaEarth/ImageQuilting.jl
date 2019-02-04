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

function convdist(img::AbstractArray, kern::AbstractArray;
                  weights::AbstractArray=fill(1.0, size(kern)))
  # choose among imfilter implementations
  imfilter_impl = get_imfilter_impl(GPU)

  wkern = weights.*kern

  A² = imfilter_impl(img.^2, weights)
  AB = imfilter_impl(img, wkern)
  B² = sum(abs2, wkern)

  D = abs.(A² .- 2AB .+ B²)

  parent(D) # always return a plain simple array
end

function genpath(extent::Dims{N}, kind::Symbol, datainds=[]) where {N}
  path = Vector{Int}()

  if kind == :raster
    for ind in LinearIndices(extent)
      push!(path, ind)
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
      dilated = dilate(grid)
      append!(path, findall(vec(dilated .& .!grid)))
      grid = dilated
    end
  end

  if kind == :data
    @assert !isempty(datainds) "data path cannot be generated without data"

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
