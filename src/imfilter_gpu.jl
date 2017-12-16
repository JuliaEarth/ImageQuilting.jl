## Copyright (c) 2015, Júlio Hoffimann Mendes <juliohm@stanford.edu>
##
## Permission to use, copy, modify, and/or distribute this software for any
## purpose with or without fee is hereby granted, provided that the above
## copyright notice and this permission notice appear in all copies.
##
## THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
## WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
## MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
## ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
## WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
## ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
## OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

function imfilter_gpu{T<:Real,K<:Real,N}(img::AbstractArray{T,N}, kern::AbstractArray{K,N}, border::AbstractString)
  # GPU metadata
  ctx = GPU.ctx; queue = GPU.queue
  mult_kernel = GPU.mult_kernel

  # operations with complex type
  img = map(Complex64, img)
  kern = map(Complex64, kern)

  # Int prefix is a workaround for julia #15276
  prepad  = Int[div(size(kern,i)-1, 2) for i = 1:N]
  postpad = Int[div(size(kern,i),   2) for i = 1:N]
  fullpad = Int[nextprod([2,3], size(img,i) + prepad[i] + postpad[i]) - size(img, i) - prepad[i] for i = 1:N]

  A = border == "inner" ? img : padarray(img, Pad(Symbol(border), prepad, fullpad))
  A = parent(A)

  # OpenCL FFT expects powers of 2, 3, 5, 7, 11 or 13
  paddings = clfftpad(A)
  A = padarray(A, Pad(:symmetric, zeros(Int, ndims(A)), paddings))
  A = parent(A)

  krn = zeros(Complex64, size(A))
  indexesK = ntuple(d->[size(krn,d)-prepad[d]+1:size(krn,d);1:size(kern,d)-prepad[d]], N)
  krn[indexesK...] = reflect(kern)

  # plan FFT
  p = clfft.Plan(Complex64, ctx, size(A))
  clfft.set_layout!(p, :interleaved, :interleaved)
  clfft.set_result!(p, :inplace)
  clfft.bake!(p, queue)

  # populate GPU memory
  bufA = cl.Buffer(Complex64, ctx, :copy, hostbuf=A)
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
  AF = padarray(AF, Pad(:symmetric, zeros(Int, ndims(AF)), -paddings))
  AF = parent(AF)

  if border == "inner"
    out = Array{realtype(eltype(AF))}(([size(img)...] - prepad - postpad)...)
    indexesA = ntuple(d->postpad[d]+1:size(img,d)-prepad[d], N)
    copyreal!(out, AF, indexesA)
  else
    out = Array{realtype(eltype(AF))}(size(img))
    indexesA = ntuple(d->postpad[d]+1:size(img,d)+postpad[d], N)
    copyreal!(out, AF, indexesA)
  end

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
