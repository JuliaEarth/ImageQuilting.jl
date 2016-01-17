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

function tau_model(events::AbstractVector{Int}, D₁::AbstractArray, Dₙ::AbstractArray)
  nevents = length(events)

  # trivial mass function
  nevents == 1 && return ones(1)

  # sources of information: primary + auxiliary₁ + auxiliary₂ + ...
  nsources = 1 + length(Dₙ)

  # distance matrix
  D = zeros(nevents, nsources)
  D[:,1] = D₁[events]
  for j=2:nsources
    D[:,j] = Dₙ[j-1][events]
  end

  # conditional probabilities
  P = exp(-D)
  P = broadcast(/, P, sum(P, 1))

  # prior to data all events are equally probable
  x₀ = (1 - 1/nevents) / (1/nevents)

  # assume no redundancy
  X = (1 - P) ./ P
  x = x₀ * prod(X/x₀, 2)

  p = 1 ./ (1 + x)
end
