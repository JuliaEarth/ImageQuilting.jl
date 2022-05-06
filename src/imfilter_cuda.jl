# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_device(device::CUDALibs{R},
                         img::AbstractArray{T,N},
                         kern::AbstractArray{K,N}) where {R, T<:Real,K<:Real,N}

    img_gpu_fft = nothing
    kern_gpu_fft = nothing

    size_all = Tuple(map(t->t[1]+t[2]-1,zip(size(img),size(kern))))

    img_gpu = something

    @sync begin

        Threads.@spawn begin
            img_gpu = CUDA.zeros(T,size_all)
            img_indexes = ntuple(d->let pad_size = div(size(kern,d)-1,2); (pad_size+1):size(img,d)+pad_size end, N)
            img_gpu[img_indexes...] = img
        end

        kern_gpu = CUDA.zeros(T,size_all)
        kern_indexes = axes(kern)
        kern_gpu[kern_indexes...] = kern;
        kern_gpu_fft = CUFFT.fft(kern_gpu); kern_gpu = nothing
    end

    img_gpu_fft = CUFFT.fft(img_gpu); img_gpu = nothing

    kern_gpu_fft = conj.(kern_gpu_fft)

    out = img_gpu_fft .* kern_gpu_fft; img_gpu_fft = nothing; kern_gpu_fft = nothing

    out = CUFFT.ifft(out)

    # remove padding
    ix = map(x -> Base.OneTo(x), size(img) .- size(kern) .+ 1)
    out = out[ix...]

    out = map(real, out)
    out = Array(out)

    out
end