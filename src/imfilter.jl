# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------
@platform parameter clear
@platform parameter accelerator_count
@platform parameter accelerator_api

@platform default function which_platform()
  println("Running on DEFAULT PLATFORM")
end

@platform aware function which_platform({accelerator_count::(@atleast 1), accelerator_api::CUDA_API})
  println("Running on CUDA GPU")
end

@platform default function imfilter_kernel(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end

@platform aware function imfilter_kernel({accelerator_count::(@atleast 1), accelerator_api::CUDA_API}, img, krn)
  
  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # pad images to common size
  padimg  = padarray(img, Fill(zero(T), ntuple(i->0, N), size(krn) .- 1))
  padkrn  = padarray(krn, Fill(zero(T), ntuple(i->0, N), size(img) .- 1))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftimg = padimg |> CuArray |> CUFFT.fft
  fftkrn = padkrn |> CuArray |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # unpad result
  start  = CartesianIndex(ntuple(i->1, N))
  finish = CartesianIndex(size(img) .- (size(krn) .- 1))
  real.(result[start:finish]) |> Array
end