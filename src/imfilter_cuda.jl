# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_resource(resource::CUDALibs, img, kern)

  sizeall = size(img) .+ size(kern) .- 1

  N = ndims(img)
  T = eltype(img)

  imggpu = Threads.@spawn begin
    padimg = CUDA.zeros(T,sizeall)
    indices = ntuple(d->let padsize = div(size(kern,d)-1,2); (padsize+1):size(img,d)+padsize end, N)
    padimg[indices...] = img
    padimg
  end

  padkern = CUDA.zeros(T,sizeall)
  kernindices = axes(kern)
  padkern[kernindices...] = kern

  fftkern = conj.(CUFFT.fft(padkern))
  fftimg = CUFFT.fft(fetch(imggpu))

  out = CUFFT.ifft(fftimg .* fftkern)

  # remove padding
  ix = map(x -> Base.OneTo(x), size(img) .- size(kern) .+ 1)
  out = out[ix...]

  out = Array(real.(out))

  out
end