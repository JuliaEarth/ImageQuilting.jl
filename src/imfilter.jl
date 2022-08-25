# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end

function imfilter_gpu(img, krn)
  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # pad kernel to common size with image
  padsize = size(img) .- size(krn)
  padkrn  = padarray(krn, Fill(zero(T), ntuple(i->0, N), padsize))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftimg = img |> CUFFT.fft
  fftkrn = padkrn |> CuArray |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # recover result
  finalsize = size(img) .- (size(krn) .- 1)
  real.(result[CartesianIndices(finalsize)]) |> Array
end

const imfilter_kernel = CUDA.functional() ? imfilter_gpu : imfilter_cpu
