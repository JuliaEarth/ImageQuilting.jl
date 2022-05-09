# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_resource(resource::CUDALibs{R},
                         img::AbstractArray{T,N},
                         kern::AbstractArray{K,N}) where {R, T<:Real,K<:Real,N}

  sizeall = Tuple(map(t->t[1]+t[2]-1,zip(size(img),size(kern))))

  imggpu = nothing
  kerngpu = nothing

  @sync begin
    Threads.@spawn begin
      imggpu = CUDA.zeros(T,sizeall)
      imgindexes = ntuple(d->let padsize = div(size(kern,d)-1,2); (padsize+1):size(img,d)+padsize end, N)
      imggpu[imgindexes...] = img
    end

    kerngpu = CUDA.zeros(T,sizeall)
    kernindexes = axes(kern)
    kerngpu[kernindexes...] = kern
    kerngpu = CUFFT.fft(kerngpu)
    kerngpu = conj.(kerngpu)
  end

  imggpu = CUFFT.fft(imggpu)

  out = imggpu .* kerngpu

  out = CUFFT.ifft(out)
 
  # remove padding
  out = let ix = map(x -> Base.OneTo(x), size(img) .- size(kern) .+ 1); out[ix...] end

  out = real.(out)
  out = Array(out)

  out
end