using CUDA
using CUDA.CUFFT
#using FFTW
using BenchmarkTools
using ImageFiltering
#using DSP


function imfilter_gpu(img::AbstractArray{T,N},
                      kern::AbstractArray{K,N}) where {T<:Real,K<:Real,N}

    img_gpu_fft = nothing
    kern_gpu_fft = nothing

    size_all = Tuple(map(t->t[1]+t[2]-1,zip(size(img),size(kern))))

    @sync begin

        Threads.@spawn begin
            img_gpu = CUDA.zeros(T,size_all)
            img_indexes = ntuple(d->let pad_size = div(size(kern,d)-1,2); (pad_size+1):size(img,d)+pad_size end, N)
            img_gpu[img_indexes...] = img
        end

        kern_gpu = CUDA.zeros(T,size_all)
        kern_indexes = axes(kern)
        kern_gpu[kern_indexes...] = kern;
        kern_gpu_fft = fft(kern_gpu); kern_gpu = nothing
    end

    img_gpu_fft = fft(img_gpu); img_gpu = nothing

    kern_gpu_fft = conj.(kern_gpu_fft)

    out = img_gpu_fft .* kern_gpu_fft; img_gpu_fft = nothing; kern_gpu_fft = nothing

    out = ifft(out)

    # remove padding
    out = out[axes(img)...]

    out = map(real, out)
    out = Array(out)

    out
end


println("start")

#img = rand(Float32,(512,256,256))
#kern = rand(Float32,(16,16,16))

img = rand(Float32,(256,256))
kern = rand(Float32,(32,32))

# Create a two-dimensional discrete unit impulse function.
#img =  fill(0,(9,9)); # rand(Int32,(9,9));
#img[5,5] = 1;

# Specify a filter coefficient mask and set the center of the mask as the origin.
#kern = centered([1 2 3 ; 4 5 6 ; 7 8 9]);

println("FIRST PASS")
r1 =  @btime imfilter_gpu(img, kern)
println(r1[128,128])
#println(Integer.(round.(real(r1))))

println("SECOND PASS - I - corr")
r21 = @btime imfilter(img, kern, Fill(0, kern))
println(r21[128,128])
#println(r21)

#println("SECOND PASS - II - conv")
#r22 = @btime imfilter(img, reflect(kern), Fill(0, kern))
#println(r22[128,128,128])
#println(r22)

#println("THIRD PASS")
#r3 = @btime conv(img, kern)
#println(r3[128,128,128])
#println(r3)

#pad_kern = similar(img)
#pad_kern[axes(kern)...] = kern

#println("FOURTH PASS")
#r4 = @btime ifft(fft(img) .* conj.(fft(pad_kern)))
#println(r4[128,128,128])
