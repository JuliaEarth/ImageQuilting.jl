# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_gpu(img::AbstractArray{T,N},
                      kern::AbstractArray{K,N}) where {T<:Real,K<:Real,N}
  # GPU metadata
  ctx = GPU.ctx; queue = GPU.queue
  mult_kernel = GPU.mult_kernel

  # operations with complex type
  img  = map(Complex64, img)
  kern = map(Complex64, kern)

  # kernel may require padding
  prepad  = ntuple(d->(size(kern,d)-1) ÷ 2, N)
  postpad = ntuple(d->(size(kern,d)  ) ÷ 2, N)

  # OpenCL FFT expects powers of 2, 3, 5, 7, 11 or 13
  clpad = clfftpad(img)
  A = padarray(img, Pad(:symmetric, zeros(Int, ndims(img)), clpad))
  A = parent(A)

  krn = zeros(Complex64, size(A))
  indexesK = ntuple(d->[size(A,d)-prepad[d]+1:size(A,d);1:size(kern,d)-prepad[d]], N)
  krn[indexesK...] = reflect(kern)

  # plan FFT
  p = clfft.Plan(Complex64, ctx, size(A))
  clfft.set_layout!(p, :interleaved, :interleaved)
  clfft.set_result!(p, :inplace)
  clfft.bake!(p, queue)

  # populate GPU memory
  bufA   = cl.Buffer(Complex64, ctx, :copy, hostbuf=A)
  bufkrn = cl.Buffer(Complex64, ctx, :copy, hostbuf=krn)
  bufRES = cl.Buffer(Complex64, ctx, length(A))

  # compute ifft(fft(A).*fft(kern))
  clfft.enqueue_transform(p, :forward, [queue], bufA, nothing)
  clfft.enqueue_transform(p, :forward, [queue], bufkrn, nothing)
  queue(mult_kernel, length(A), nothing, bufA, bufkrn, bufRES)
  clfft.enqueue_transform(p, :backward, [queue], bufRES, nothing)

  # get result back
  AF = reshape(cl.read(queue, bufRES), size(A))

  # undo OpenCL FFT paddings
  AF = view(AF, ntuple(d->1:size(AF,d)-clpad[d], N)...)

  out = Array{realtype(eltype(AF))}(ntuple(d->size(img,d) - prepad[d] - postpad[d], N))
  indexesA = ntuple(d->postpad[d]+1:size(img,d)-prepad[d], N)
  copyreal!(out, AF, indexesA)

  out
end

@generated function reflect{T,N}(A::AbstractArray{T,N})
    quote
        B = Array{T}(size(A))
        @nexprs $N d->(n_d = size(A, d)+1)
        @nloops $N i A d->(j_d = n_d - i_d) begin
            @nref($N, B, j) = @nref($N, A, i)
        end
        B
    end
end

for N = 1:5
    @eval begin
        function copyreal!{T<:Real}(dst::Array{T,$N}, src, I::Tuple{Vararg{UnitRange{Int}}})
            @nexprs $N d->(I_d = I[d])
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = real(@nref $N src j)
            end
            dst
        end
        function copyreal!{T<:Complex}(dst::Array{T,$N}, src, I::Tuple{Vararg{UnitRange{Int}}})
            @nexprs $N d->I_d = I[d]
            @nloops $N i dst d->(j_d = first(I_d)+i_d-1) begin
                (@nref $N dst i) = @nref $N src j
            end
            dst
        end
    end
end

realtype{R<:Real}(::Type{R}) = R
realtype{R<:Real}(::Type{Complex{R}}) = R
