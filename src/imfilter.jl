# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end

function imfilter_cuda(img, krn)
  # retrieve basic info
  N = ndims(img)
  T = eltype(img)

  # pad kernel to common size with image
  padkrn = CUDA.zeros(Float32, size(img))
  copyto!(padkrn, CartesianIndices(krn), array_gpu(krn), CartesianIndices(krn))

  # perform ifft(fft(img) .* conj.(fft(krn)))
  fftimg = img |> CUFFT.fft
  fftkrn = padkrn |> CUFFT.fft
  result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft

  # recover result
  finalsize = size(img) .- (size(krn) .- 1)
  real.(result[CartesianIndices(finalsize)]) |> Array
end

function imfilter_opencl(img, krn)
  # retrieve basic info
  N = ndims(img)
  T = ComplexF64

  # retrieve OpenCL info
  device, ctx, queue = cl.create_compute_context()

  # build OpenCL program kernels
  conj_kernel = build_conj_kernel(ctx)
  mult_kernel = build_mult_kernel(ctx)

  # pad krn to common size with img
  padkrn = zeros(eltype(img), size(img))
  padkrn[CartesianIndices(krn)] = krn

  # convert to Complex
  fftimg = T.(img)
  fftkrn = T.(padkrn)

  # OpenCl setup
  plan = CLFFT.Plan(T, ctx, size(img))
  CLFFT.set_layout!(plan, :interleaved, :interleaved)
  CLFFT.set_result!(plan, :inplace)
  CLFFT.bake!(plan, queue)

  # populate device memory
  bufimg = cl.Buffer(T, ctx, :copy, hostbuf=fftimg)
  bufkrn = cl.Buffer(T, ctx, :copy, hostbuf=fftkrn)
  bufresult = cl.Buffer(T, ctx, :w, length(img))

  # transform img and krn to FFT representation 
  CLFFT.enqueue_transform(plan, :forward, [queue], bufimg, nothing)
  CLFFT.enqueue_transform(plan, :forward, [queue], bufkrn, nothing)

  # compute ifft(fft(A).*conj.(fft(krn)))
  queue(conj_kernel, length(img), nothing, bufkrn)
  queue(mult_kernel, length(img), nothing, bufimg, bufkrn, bufresult)
  CLFFT.enqueue_transform(plan, :backward, [queue], bufresult, nothing)

  # recover result
  result = reshape(cl.read(queue, bufresult), size(img))
  real_result = real.(result)

  finalsize = size(img) .- (size(krn) .- 1)
  real_result[CartesianIndices(finalsize)]
end

if selected_imfilter_algorithm == :CUDA
  const imfilter_kernel = imfilter_cuda
elseif selected_imfilter_algorithm == :OpenCL
  const imfilter_kernel = imfilter_opencl
elseif selected_imfilter_algorithm == :CPU
  const imfilter_kernel = imfilter_cpu
end