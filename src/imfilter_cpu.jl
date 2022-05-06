# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_device(device,
                         img::AbstractArray{T,N},
                         kern::AbstractArray{K,N}) where {R,T<:Real,K<:Real,N}
  imfilter(img, centered(kern), Inner(), Algorithm.FFT())
end
