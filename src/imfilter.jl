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

  # pad images to common size
  imgsize = size(img)
  krnsize = size(krn)
  padimg  = padarray(img, Fill(zero(T), ntuple(i->0, N), krnsize .- 1))
  padkrn  = padarray(krn, Fill(zero(T), ntuple(i->0, N), imgsize .- 1))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftimg = padimg |> CuArray |> CUFFT.fft
  fftkrn = padkrn |> CuArray |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # unpad result
  start  = CartesianIndex(ntuple(i->1, N))
  finish = CartesianIndex(imgsize .- (krnsize .- 1))
  real.(result[start:finish]) |> Array
end

const imfilter_kernel = CUDA.functional() ? imfilter_gpu : imfilter_cpu
