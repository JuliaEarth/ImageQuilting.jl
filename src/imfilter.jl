# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end

# preallocated memory for krn optimization on GPU
krnbuffer_gpu = nothing

function alloc_krn_buffer_gpu!(sz)
  global krnbuffer_gpu
  
  if isnothing(krnbuffer_gpu) || size(krnbuffer_gpu) != sz
    krnbuffer_gpu = CuArray(zeros(sz))
  end

  krnbuffer_gpu
end

function imfilter_gpu(img, krn)
  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # copy krn data to preallocated GPU memory
  paddedkrn = alloc_krn_buffer_gpu!(size(img))
  copyto!(paddedkrn, CartesianIndices(krn), CuArray(krn), CartesianIndices(krn))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftimg = img |> CUFFT.fft
  fftkrn = paddedkrn |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # recover result
  finalsize = size(img) .- (size(krn) .- 1)
  real.(result[CartesianIndices(finalsize)]) |> Array
end

const imfilter_kernel = CUDA.functional() ? imfilter_gpu : imfilter_cpu
