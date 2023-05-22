# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function relaxation(distance, auxdistances, cutoff)
  # patterns enabled in the training image
  enabled = .!isinf.(distance)
  npatterns = sum(enabled)

  # candidates with good overlap
  dbsize = all(distance[enabled] .== 0) ? npatterns : ceil(Int, cutoff * npatterns)
  overlapdb = partialsortperm(vec(distance), 1:dbsize)

  # candidates in accordance with auxiliary data
  naux = length(auxdistances)
  softdb = [Vector{Int}() for i in 1:naux]

  patterndb = []
  softdistance = [copy(auxdistances[i]) for i in 1:naux]
  frac = 0.1 * (dbsize / npatterns)
  while true
    softdbsize = ceil(Int, frac * npatterns)

    patterndb = overlapdb
    for n in 1:naux
      softdistance[n][softdb[n]] .= Inf
      softdb[n] = [softdb[n]; partialsortperm(vec(softdistance[n]), 1:(softdbsize - length(softdb[n])))]

      patterndb = fastintersect(patterndb, softdb[n], length(distance))

      isempty(patterndb) && break
    end

    !isempty(patterndb) && break
    frac = min(frac + 0.1, 1)
  end

  patterndb
end

function fastintersect(A, B, nbits)
  bitsA = falses(nbits)
  bitsB = falses(nbits)
  bitsA[A] .= true
  bitsB[B] .= true

  findall(bitsA .& bitsB)
end
