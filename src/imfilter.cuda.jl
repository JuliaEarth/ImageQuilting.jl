# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

# using the CUDA kernel
const array_kernel(array, ::CUDAKernel) = CuArray{Float32}(array)
const view_kernel(array, I, ::CUDAKernel) = Array(array[I])

function imfilter_kernel(img, krn, ::CUDAKernel)
  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # pad kernel to common size with image
  padkrn = CUDA.zeros(Float32, size(img))
  copyto!(padkrn, CartesianIndices(krn), array_gpu(krn), CartesianIndices(krn))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftimg = img |> CUFFT.fft
  fftkrn = padkrn |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # recover result
  finalsize = size(img) .- (size(krn) .- 1)
  real.(result[CartesianIndices(finalsize)]) |> Array
end

