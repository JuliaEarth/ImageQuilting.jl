## Copyright (c) 2014 Júlio Hoffimann Mendes
##
## This file is part of ImageQuilting.
##
## ImageQuilting is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ImageQuilting is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ImageQuilting.  If not, see <http://www.gnu.org/licenses/>.
##
## Created: 12 Aug 2014
## Author: Júlio Hoffimann Mendes

using Images
try
    require("ImageView") # throw an error if not available
    global view = Main.ImageView.view
catch
    # replace view() by nothing
    global view(args...; kargs...) = (nothing, nothing)
end


function imquilt(img::Image, args...; kargs...)
    simg = separate(img)
    X, props = data(simg), properties(simg)
    return Image(imquilt(X, args...; kargs...), props)
end


## Image Quilting
##
## Synthetisize an image by stitching together small patches
## from a 2D training image.
##
##   img   - 2D training image (Grayscale or RGB)
##   sizeₜ - tile size
##   nₜ    - number of tiles to stitch together
##   tol   - tolerance used for finding best tiles
##   show  - whether to show the output image or not
##
## EFROS, A.; FREEMAN, W. T., 2001. Image Quilting for
## Texture Synthesis and Transfer.
function imquilt(img::AbstractArray, sizeₜ::Integer, nₜ::Integer; tol=1e-3, show=false)
    @assert sizeₜ ≥ 12 "tile size must be at least 12"

    ## Enforce the format to be:
    ##   Grayscale => size(X) = (m,n,1)
    ##   RGB       => size(X) = (m,n,3)
    X = float64(img)
    X = ndims(X) == 3 && size(X, 1) == 3 ? permutedims(X, [2 3 1]) : X
    X = ndims(X) == 2 ? repeat(X, outer=[1 1 1]) : X
    mₓ, nₓ, nlayers = size(X)

    overlap = sizeₜ ÷ 6
    npixels = nₜ * sizeₜ - (nₜ-1) * overlap
    Y = zeros(npixels, npixels, nlayers)

    show && ((canvas, _) = view(colorim(Y)))

    # scan the output image tile by tile
    for i=1:nₜ, j=1:nₜ
        # tile corners are given by (iₛ,jₛ) and (iₑ,jₑ)
        iₛ = (i-1)sizeₜ - (i-1)overlap + 1
        jₛ = (j-1)sizeₜ - (j-1)overlap + 1
        iₑ = iₛ + sizeₜ - 1
        jₑ = jₛ + sizeₜ - 1

        Tᵧ = Y[iₛ:iₑ,jₛ:jₑ,:]

        # compute the distance between the target Tᵧ and
        # all possible tiles Tₓ in the training image
        distance = zeros(mₓ-sizeₜ+1, nₓ-sizeₜ+1)
        if j > 1
            D = convdist(X, Tᵧ[:,1:overlap,:])
            distance += D[:,1:nₓ-sizeₜ+1]
        end
        if i > 1
            D = convdist(X, Tᵧ[1:overlap,:,:])
            distance += D[1:mₓ-sizeₜ+1,:]
        end
        if i > 1 && j > 1
            D = convdist(X, Tᵧ[1:overlap,1:overlap,:])
            distance -= D[1:mₓ-sizeₜ+1,1:nₓ-sizeₜ+1]
        end
        distance = abs(distance) # amend floating point weirdness

        # pick a candidate at random from the bag of best tiles
        bag = find(distance .<= (1+tol)minimum(distance))
        idx = bag[rand(1:length(bag))]
        iᵦ, jᵦ = ind2sub(size(distance), idx)

        Tᵦ = X[iᵦ:iᵦ+sizeₜ-1,jᵦ:jᵦ+sizeₜ-1,:]

        # error surface for vertical and horizontal overlap in
        # the first channel of the image (i.e. "R" in "RGB")
        ev = (Tᵧ[:,1:overlap,1] - Tᵦ[:,1:overlap,1]).^2
        eh = (Tᵧ[1:overlap,:,1] - Tᵦ[1:overlap,:,1]).^2

        # minimum boundary cut mask
        m, n, _ = size(Tᵧ)
        M = falses(m, n)
        j > 1 && (M[:,1:overlap] |= mincut(ev))
        i > 1 && (M[1:overlap,:] |= mincut(eh')')

        # paste contributions from Tᵧ and Tᵦ
        Y[iₛ:iₑ,jₛ:jₑ,:] = M.*Tᵧ + !M.*Tᵦ

        show && view(canvas, colorim(Y))
    end

    # remove ghost dimension and permute back
    Y = size(Y, 3) == 3 ? permutedims(Y, [3 1 2]) : Y
    Y = size(Y, 3) == 1 ? Y[:,:,1] : Y

    return convert(typeof(img), Y)
end


## Minimum Boundary Cut
##
## Given the error surface for a vertical overlap,
## compute the mask of minimum error cut:
##
##                  111/00
##                  11/000
##                  1|0000
##                  1|0000
##                  11\000
##                  1111\0
##
## For a horizontal overlap, take the transpose of
## the input surface and output mask.
function mincut(ev::Matrix)
    E, M = ev, falses(size(ev))

    # forward accumulation
    nrow, ncol = size(E)
    for i=2:nrow
        E[i,1] = ev[i,1] + minimum(ev[i-1,1:2])
        for j=2:ncol-1
            E[i,j] = ev[i,j] + minimum(ev[i-1,j-1:j+1])
        end
        E[i,end] = ev[i,end] + minimum(ev[i-1,end-1:end])
    end

    # backward search
    _, idx = findmin(E[end,:])
    M[end,1:idx] = true
    for i=nrow-1:-1:1
        for j=1:ncol
            if idx < ncol && minimum(E[i,max(idx-1,1):idx+1]) == E[i,idx+1]
                idx += 1
            elseif idx > 1 && minimum(E[i,idx-1:min(idx+1,ncol)]) == E[i,idx-1]
                idx -= 1
            end
        end
        M[i,1:idx] = true
    end

    return M
end


function convdist(X::AbstractArray, mask::AbstractArray)
    m, n, nlayers = size(mask)

    @parallel (+) for k=1:nlayers
        A = X[:,:,k]
        B = mask[:,:,k]

        A² = imfilter_fft(A.^2, ones(m, n), "inner")
        AB = imfilter_fft(A, B, "inner")
        B² = sum(B.^2)

        A² - 2AB + B²
    end
end
