## Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
##
## Permission to use, copy, modify, and/or distribute this software for any
## purpose with or without fee is hereby granted, provided that the above
## copyright notice and this permission notice appear in all copies.
##
## THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
## WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
## ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
## WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
## ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
## OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

function get_imfilter_impl(GPU)
  if GPU ≠ nothing
    imfilter_gpu
  else
    imfilter_fft
  end
end

function convdist(Xs::AbstractArray, masks::AbstractArray; weights=nothing, inner=true)
  padding = inner == true ? "inner" : "symmetric"

  # choose among imfilter implementations
  imfilter_impl = get_imfilter_impl(GPU)

  result = []
  for (X, mask) in zip(Xs, masks)
    weights = weights ≠ nothing ? weights : ones(mask)

    A² = imfilter_impl(X.^2, weights.*ones(mask), padding)
    AB = imfilter_impl(X, weights.*mask, padding)
    B² = sum((weights.*mask).^2)

    push!(result, abs(A² - 2AB + B²))
  end

  sum(result)
end

function genpath(extent::NTuple{3,Integer}, kind::Symbol, datum=Set())
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
