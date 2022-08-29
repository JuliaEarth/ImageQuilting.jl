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

@platform aware function which_platform({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API})
  println("Running on OpenCL GPU")
end

@platform default function imfilter_kernel(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end


@platform aware function imfilter_kernel({accelerator_count::(@atleast 1), accelerator_api::CUDA_API}, img, krn)

   # retrieve basic info
   N = ndims(img)
   T = eltype(img)
 
   # pad kernel to common size with image
   padkrn = CUDA.zeros(size(img))
   copyto!(padkrn, CartesianIndices(krn), CuArray(krn), CartesianIndices(krn))
 
   # perform ifft(fft(img) .* conj.(fft(krn)))
   fftimg = img |> CUFFT.fft
   fftkrn = padkrn |> CuArray |> CUFFT.fft
   result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft
 
   # recover result
   finalsize = size(img) .- (size(krn) .- 1)
   real.(result[CartesianIndices(finalsize)]) |> Array
   
 end


const GPU = gpu_setup()

@platform aware function imfilter_kernel({accelerator_count::(@atleast 1), accelerator_api::OpenCL_API}, img, kern)
  

   # retrieve basic info
   N = ndims(img)
   T = ComplexF64

   # GPU metadata
   ctx = GPU.ctx; queue = GPU.queue
   mult_kernel = GPU.mult_kernel
 
   # operations with complex type
   img  = T.(img)
   kern = T.(kern)
   
   # kernel may require padding
   prepad  = ntuple(d->(size(kern,d)-1) ÷ 2, N)
   postpad = ntuple(d->(size(kern,d)  ) ÷ 2, N)
 
   # OpenCL FFT expects powers of 2, 3, 5, 7, 11 or 13
   clpad = clfftpad(img)
   A = padarray(img, Pad(:symmetric, zeros(Int, ndims(img)), clpad))
   A = parent(A)
 
   krn = zeros(T, size(A))
   indexesK = ntuple(d->[size(A,d)-prepad[d]+1:size(A,d);1:size(kern,d)-prepad[d]], N)
   krn[indexesK...] = reflect(kern) 

   # plan FFT
   p = clfft.Plan(T, ctx, size(A))
   clfft.set_layout!(p, :interleaved, :interleaved)
   clfft.set_result!(p, :inplace)
   clfft.bake!(p, queue)
 
   # populate GPU memory
   bufA   = cl.Buffer(T, ctx, :copy, hostbuf=A)
   bufkrn = cl.Buffer(T, ctx, :copy, hostbuf=krn)
   bufRES = cl.Buffer(T, ctx, length(A))
   
   # compute ifft(fft(A).*fft(kern))
   clfft.enqueue_transform(p, :forward, [queue], bufA, nothing)
   clfft.enqueue_transform(p, :forward, [queue], bufkrn, nothing)
   queue(mult_kernel, length(A), nothing, bufA, bufkrn, bufRES)

   clfft.enqueue_transform(p, :backward, [queue], bufRES, nothing)

   # get result back
   AF = reshape(cl.read(queue, bufRES), size(A))
 
   # undo OpenCL FFT paddings
   AF = view(AF, ntuple(d->1:size(AF,d)-clpad[d], N)...)
  
   out = Array{realtype(eltype(AF))}(undef, ntuple(d->size(img,d) - prepad[d] - postpad[d], N)...)
   indexesA = ntuple(d->postpad[d]+1:size(img,d)-prepad[d], N)
   copyreal!(out, AF, indexesA)
   
   GC.gc()

   out
  end

 @generated function reflect(A::AbstractArray{T,N}) where {T,N}
    quote
        B = Array{T,N}(undef,size(A))
        @nexprs $N d->(n_d = size(A, d)+1)
        @nloops $N i A d->(j_d = n_d - i_d) begin
            @nref($N, B, j) = @nref($N, A, i)
        end
        B
    end
 end


for N = 1:5
    @eval begin
        function copyreal!(dst::Array{T,$N}, src, I::Tuple{Vararg{UnitRange{Int}}}) where {T<:Real}
            @nexprs $N d->(I_d = I[d])
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = real(@nref $N src j)
            end
            dst
        end
        function copyreal!(dst::Array{T,$N}, src, I::Tuple{Vararg{UnitRange{Int}}}) where {T<:Complex}
            @nexprs $N d->I_d = I[d]
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = @nref $N src j
            end
            dst
        end
    end
end

realtype(::Type{R}) where {R<:Real} = R
realtype(::Type{Complex{R}}) where {R<:Real} = R


@platform default function placeimg(img, soft)
   img, map(soft) do (aux, auxTI)
    auxTI 
   end 
end

@platform aware function placeimg({accelerator_count::(@atleast 1), accelerator_api::CUDA_API}, img, soft)
   img_gpu = img .|> Float32 |> CuArray

   soft_gpu = map(soft) do (aux, auxTI)
    auxTI .|> Float32 |> CuArray
   end

   img_gpu, soft_gpu
end

