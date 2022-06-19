# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

struct GPUmeta
  dev
  ctx
  queue
  mult_kernel
end

function gpu_setup()
  @assert cl ≠ nothing "OpenCL.jl not installed, cannot use GPU"
  @assert clfft ≠ nothing "CLFFT.jl not installed, cannot use GPU"

  devs = cl.devices(:gpu)
  if isempty(devs)
    @warn "GPU not found, falling back to other OpenCL devices"
    devs = cl.devices()
  end
  @assert !isempty(devs) "OpenCL device not found, make sure drivers are installed"

  dev = []
  devnames = map(d -> d[:platform][:name], devs)
  for vendor in ["NVIDIA","AMD","Intel"], (idx,name) in enumerate(devnames)
    if occursin(vendor, name)
      dev = devs[idx]
      break
    end
  end

  devtype = uppercase(string(dev[:device_type]))
  devname = dev[:name]

  @info "using $devtype $devname"

  ctx = cl.Context(dev)
  queue = cl.CmdQueue(ctx)
  mult_kernel = clkernels(ctx)

  GPUmeta(dev, ctx, queue, mult_kernel)
end

function clkernels(ctx)
  mult_kernel = "
    __kernel void mult(__global const float2 *a,
                       __global const float2 *b,
                       __global float2 *c)
    {
      int gid = get_global_id(0);
      c[gid].x = a[gid].x*b[gid].x - a[gid].y*b[gid].y;
      c[gid].y = a[gid].x*b[gid].y + a[gid].y*b[gid].x;
    }
  "

  # build OpenCL program
  prog = cl.Program(ctx, source=mult_kernel) |> cl.build!

  cl.Kernel(prog, "mult")
end

function clfftpad(A::AbstractArray)
  # clFFT releases support powers of 2, 3, 5, ...
  radices = [2,3,5]
  v = clfft.version()
  v ≥ v"2.8.0"  && push!(radices, 7)
  v ≥ v"2.12.0" && push!(radices, 11, 13)

  result = Int[]
  for s in size(A)
    fs = keys(factor(s))
    if fs ⊆ radices
      push!(result, 0)
    else
      # Try a closer number that has prime factors of 2 and 3.
      # Use the next power of 2 (say N) to get multiple new
      # candidates.
      N = nextpow2(s)

      # fractions of N: 100%, 93%, 84%, 75%, 56%
      candidates = [N, 15(N÷16), 27(N÷32), 3(N÷4), 9(N÷16)]
      candidates = candidates[candidates .> s]
      n = minimum(candidates)

      push!(result, n - s)
    end
  end

  result
end

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
