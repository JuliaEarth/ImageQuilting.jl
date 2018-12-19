# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu(img::AbstractArray{T,N},
                      kern::AbstractArray{K,N}) where {T<:Real,K<:Real,N}
  imfilter(img, centered(kern), Inner(), Algorithm.FFT())
end
