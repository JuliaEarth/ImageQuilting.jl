# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

const array_kernel(::OpenCLMethod, array) = array

const view_kernel(::OpenCLMethod, array, I) = view(array, I)

function imfilter_kernel(::OpenCLMethod, img, krn)
  # retrieve basic info
  T = ComplexF64

  # retrieve OpenCL info
  device, ctx, queue = cl.create_compute_context()

  # build OpenCL program kernels
  conj_kernel = build_conj_kernel(ctx)
  mult_kernel = build_mult_kernel(ctx)

  # pad img to support CLFFT operations
  padimg = pad_opencl_img(img)

  # pad krn to common size with padimg
  padkrn = zeros(eltype(img), size(padimg))
  padkrn[CartesianIndices(krn)] = krn

  # convert to Complex
  fftimg = T.(padimg)
  fftkrn = T.(padkrn)

  # OpenCl setup
  plan = CLFFT.Plan(T, ctx, size(fftimg))
  CLFFT.set_layout!(plan, :interleaved, :interleaved)
  CLFFT.set_result!(plan, :inplace)
  CLFFT.bake!(plan, queue)

  # populate device memory
  bufimg = cl.Buffer(T, ctx, :copy, hostbuf=fftimg)
  bufkrn = cl.Buffer(T, ctx, :copy, hostbuf=fftkrn)
  bufres = cl.Buffer(T, ctx, :w, length(fftimg))

  # transform img and krn to FFT representation 
  CLFFT.enqueue_transform(plan, :forward, [queue], bufimg, nothing)
  CLFFT.enqueue_transform(plan, :forward, [queue], bufkrn, nothing)

  # compute ifft(fft(A).*conj.(fft(krn)))
  queue(conj_kernel, length(fftimg), nothing, bufkrn)
  queue(mult_kernel, length(fftimg), nothing, bufimg, bufkrn, bufres)
  CLFFT.enqueue_transform(plan, :backward, [queue], bufres, nothing)

  # recover result
  result = reshape(cl.read(queue, bufres), size(fftimg))
  real_result = real.(result)

  finalsize = size(img) .- (size(krn) .- 1)
  real_result[CartesianIndices(finalsize)]
end

function pad_opencl_img(img)
  # OpenCL FFT expects products of powers of 2, 3, 5, 7, 11 or 13
  radices = CLFFT.supported_radices()
  newsize = map(dim -> nextprod(radices, dim), size(img))
  
  padimg = zeros(eltype(img), newsize)
  padimg[CartesianIndices(img)] = img
  padimg
end

function build_mult_kernel(ctx)
  mult_kernel = "
  __kernel void mult(__global const double2 *a,
                      __global const double2 *b,
                      __global double2 *c)
  {
    int gid = get_global_id(0);
    c[gid].x = a[gid].x*b[gid].x - a[gid].y*b[gid].y;
    c[gid].y = a[gid].x*b[gid].y + a[gid].y*b[gid].x;
  }
  "
  prog = cl.Program(ctx, source=mult_kernel) |> cl.build!
  cl.Kernel(prog, "mult")
end

function build_conj_kernel(ctx)
  conj_kernel = "
  __kernel void conj(__global double2 *a)
  {
    int gid = get_global_id(0);
    a[gid].y = -a[gid].y;
  }
  "
  prog = cl.Program(ctx, source=conj_kernel) |> cl.build!
  cl.Kernel(prog, "conj")
end
