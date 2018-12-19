# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function tau_model(events::AbstractVector, D₁::AbstractArray, Dₙ::AbstractArray)
  nevents = length(events)

  # trivial mass function
  nevents == 1 && return [1.0]

  # sources of information: primary + auxiliary₁ + auxiliary₂ + ...
  nsources = 1 + length(Dₙ)

  # distance matrix
  D = zeros(nevents, nsources)
  D[:,1] = D₁[events]
  for j=2:nsources
    D[:,j] = Dₙ[j-1][events]
  end

  # convert distances to ranks
  idx = mapslices(sortperm, D, dims=1)
  for j=1:nsources
    r = 0; prevdist = -Inf
    for i in view(idx,:,j)
      D[i,j] > prevdist && (r += 1)
      prevdist = D[i,j]
      D[i,j] = r
    end
  end

  # conditional probabilities
  P = nevents .- D .+ 1
  P = broadcast(/, P, sum(P, dims=1))

  # prior to data all events are equally probable
  x₀ = (1 - 1/nevents) / (1/nevents)

  # assume no redundancy
  X = (1 .- P) ./ P
  x = x₀ * prod(X/x₀, dims=2)

  p = 1 ./ (1 .+ x)
end
