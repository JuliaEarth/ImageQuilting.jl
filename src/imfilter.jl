# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu(img, krn, krn_alloc)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end

function imfilter_gpu(img, krn, krn_alloc)
  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # copy krn data to preallocated GPU memory
  copyto!(krn_alloc, CartesianIndices(krn), CuArray(krn), CartesianIndices(krn))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftimg = img |> CUFFT.fft
  fftkrn = krn_alloc |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # recover result
  finalsize = size(img) .- (size(krn) .- 1)
  real.(result[CartesianIndices(finalsize)]) |> Array
end

const imfilter_kernel = CUDA.functional() ? imfilter_gpu : imfilter_cpu
