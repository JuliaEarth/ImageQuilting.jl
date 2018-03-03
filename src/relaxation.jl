# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function relaxation(distance::AbstractArray, auxdistances::AbstractArray, cutoff::Real)
  # patterns enabled in the training image
  enabled = .!isinf.(distance)
  npatterns = sum(enabled)

  # candidates with good overlap
  dbsize = all(distance[enabled] .== 0) ? npatterns : ceil(Int, cutoff*npatterns)
  overlapdb = selectperm(distance[:], 1:dbsize)

  # candidates in accordance with auxiliary data
  naux = length(auxdistances)
  softdb = fill(Int[], naux)

  patterndb = []
  softdistance = [copy(auxdistances[i]) for i=1:naux]
  frac = .1 * (dbsize / npatterns)
  while true
    softdbsize = ceil(Int, frac*npatterns)

    patterndb = overlapdb
    for n=1:length(softdistance)
      softdistance[n][softdb[n]] = Inf
      softdb[n] = [softdb[n]; selectperm(softdistance[n][:], 1:softdbsize-length(softdb[n]))]

      patterndb = quick_intersect(patterndb, softdb[n], length(distance))

      isempty(patterndb) && break
    end

    !isempty(patterndb) && break
    frac = min(frac + .1, 1)
  end

  patterndb
end

function quick_intersect(A::AbstractVector{Int}, B::AbstractVector{Int},
                         nbits::Integer)
  bitsA = falses(nbits)
  bitsB = falses(nbits)
  bitsA[A] = true
  bitsB[B] = true

  find(bitsA .& bitsB)
end
