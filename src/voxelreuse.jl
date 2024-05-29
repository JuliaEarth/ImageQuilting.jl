# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

"""
    voxelreuse(trainimg::AbstractArray{T,N}, tilesize::Dims{N};
               overlap::NTuple{N,Float64}=ntuple(i->1/6,N),
               nreal::Integer=10, kwargs...)

Returns the mean voxel reuse in `[0,1]` and its standard deviation.

### Notes

- The approximation gets better as `nreal` is made larger.
- Keyword arguments `kwargs` are passed to `iqsim` directly.
"""
function voxelreuse(
  trainimg::AbstractArray{T,N},
  tilesize::Dims{N};
  overlap::NTuple{N,Float64}=ntuple(i -> 1 / 6, N),
  nreal::Integer=10,
  kwargs...
) where {T,N}

  # overlap size from given percentage
  ovlsize = @. ceil(Int, overlap * tilesize)

  # elementary raster path
  ntiles = ntuple(i -> ovlsize[i] > 1 ? 2 : 1, N)

  # simulation grid dimensions
  simsize = @. ntiles * (tilesize - ovlsize) + ovlsize

  _, _, voxs = iqsim(trainimg, tilesize, simsize; overlap=overlap, nreal=nreal, debug=true, kwargs...)

  μ = mean(voxs)
  σ = std(voxs, mean=μ)

  μ, σ
end

"""
    voxelreuseplot(trainimg; [options])

Mean voxel reuse plot of `trainimg`.

## Available options:

* `tmin`    - minimum tile size (default to `7`)
* `tmax`    - maximum tile size (default to `minimum(size(trainimg))`)
* `overlap` - the percentage of overlap (default to 1/6 of tile size)
* `nreal`   - number of realizations (default to 10)
* `rng`     - random number generator (default to `Random.default_rng()`)
"""
function voxelreuseplot end
function voxelreuseplot! end
