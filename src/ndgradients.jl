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

function ndgradients(img::AbstractArray, points::AbstractVector; method=:ando3)
  extent = size(img)
  ndirs = length(extent)
  npoints = length(points)

  # smoothing weights
  weights = (method == :sobel ? [1,2,1] :
             method == :ando3 ? [.112737,.274526,.112737] :
             error("Unknown gradient method: $method"))

  # pad input image
  imgpad = padarray(img, ones(Int, ndirs), ones(Int, ndirs), "replicate")

  # gradient matrix
  G = zeros(npoints, ndirs)

  # compute gradient for all directions at specified points
  for i in 1:ndirs
    # kernel = centered difference + perpendicular smoothing
    if extent[i] > 1
      # centered difference
      idx = ones(Int, ndirs); idx[i] = 3
      kern = reshape([-1,0,1], idx...)
      # perpendicular smoothing
      for j in setdiff(1:ndirs, i)
        if extent[j] > 1
          idx = ones(Int, ndirs); idx[j] = 3
          kern = broadcast(*, kern, reshape(weights, idx...))
        end
      end

      A = zeros(kern)
      shape = size(kern)
      for (k, p) in enumerate(points)
        icenter = CartesianIndex(ind2sub(extent, p))
        i1 = CartesianIndex(tuple(ones(Int, ndirs)...))
        for ii in CartesianRange(shape)
          A[ii] = imgpad[ii + icenter - i1]
        end

        G[k,i] = sum(kern .* A)
      end
    end
  end

  G
end
