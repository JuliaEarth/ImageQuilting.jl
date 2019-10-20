# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function convdist(img::AbstractArray, kern::AbstractArray;
                  weights::AbstractArray=fill(1.0, size(kern)))
  # choose among imfilter implementations
  imfilter_impl = get_imfilter_impl(GPU)

  wkern = weights.*kern

  A² = imfilter_impl(img.^2, weights)
  AB = imfilter_impl(img, wkern)
  B² = sum(abs2, wkern)

  parent(abs.(A² .- 2AB .+ B²))
end

function overlap_distance!(distance::AbstractArray{T,N},
                           TI::AbstractArray{T,N}, simdev::AbstractArray{T,N},
                           tileind::CartesianIndex{N}, pasted::Set{CartesianIndex{N}},
                           geoconfig::NamedTuple) where {N,T<:Real}
  TIsize   = geoconfig.TIsize
  tilesize = geoconfig.tilesize
  ovlsize  = geoconfig.ovlsize
  spacing  = geoconfig.spacing

  distance .= 0
  for d=1:N
    # Cartesian index of previous and next tiles along dimension
    prev = CartesianIndex(ntuple(i -> i == d ? (tileind[d]-1) : tileind[i], N))
    next = CartesianIndex(ntuple(i -> i == d ? (tileind[d]+1) : tileind[i], N))

    # compute overlap distance with previous tile
    if ovlsize[d] > 1 && prev ∈ pasted
      oslice = ntuple(i -> i == d ? (1:ovlsize[d]) : (1:tilesize[i]), N)
      ovl = view(simdev, CartesianIndices(oslice))

      D = convdist(TI, ovl)

      ax = axes(D)
      dslice = ntuple(i -> i == d ? (1:TIsize[d]-tilesize[d]+1) : ax[i], N)
      distance .+= view(D, CartesianIndices(dslice))
    end

    # compute overlap distance with next tile
    if ovlsize[d] > 1 && next ∈ pasted
      oslice = ntuple(i -> i == d ? (spacing[d]+1:tilesize[d]) : (1:tilesize[i]), N)
      ovl = view(simdev, CartesianIndices(oslice))

      D = convdist(TI, ovl)

      ax = axes(D)
      dslice = ntuple(i -> i == d ? (spacing[d]+1:TIsize[d]-ovlsize[d]+1) : ax[i], N)
      distance .+= view(D, CartesianIndices(dslice))
    end
  end
end

function hard_distance!(distance::AbstractArray{T,N},
                        TI::AbstractArray{T,N},
                        harddev::AbstractArray{T,N},
                        hardmask::AbstractArray{Bool,N}) where {N,T<:Real}
  distance .= convdist(TI, harddev, weights=hardmask)
end

function soft_distance!(distance::AbstractArray{T,N},
                        AUXTI::AbstractArray{T,N},
                        softdev::AbstractArray{T,N}) where {N,T<:Real}
  distance .= convdist(AUXTI, softdev)
end

