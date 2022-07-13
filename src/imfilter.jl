# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end

img_to_fftimg_gpu = IdDict()

function imfilter_gpu(img, krn)
  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # get gpu fftimg
  if !haskey(img_to_fftimg_gpu, img)
    padimg = padarray(img, Fill(zero(T), ntuple(i->0, N), size(krn) .- 1))
    fftimg = padimg |> CuArray |> CUFFT.fft
    get!(img_to_fftimg_gpu, img, fftimg)
  end 
  fftimg = get(img_to_fftimg_gpu, img, Nothing) 
  
  # pad krn to common size
  padkrn = padarray(krn, Fill(zero(T), ntuple(i->0, N), size(img) .- 1))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftkrn = padkrn |> CuArray |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # unpad result
  start  = CartesianIndex(ntuple(i->1, N))
  finish = CartesianIndex(size(img) .- (size(krn) .- 1))
  real.(result[start:finish]) |> Array
end

const imfilter_kernel = CUDA.functional() ? imfilter_gpu : imfilter_cpu
