# ------------------------------------------------------------------
# Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu{T<:Real,K<:Real,N}(img::AbstractArray{T,N}, kern::AbstractArray{K,N})
  imfilter(img, centered(kern), Inner(), Algorithm.FFT())
end
