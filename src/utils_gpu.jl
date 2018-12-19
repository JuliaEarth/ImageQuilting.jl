# ------------------------------------------------------------------
# Licensed under the ISC License. See LICENCE in the project root.
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
    if contains(name, vendor)
      dev = devs[idx]
      break
    end
  end

  devtype = uppercase(string(dev[:device_type]))
  devname = dev[:name]

  info("using $devtype $devname")

  ctx = cl.Context(dev)
  queue = cl.CmdQueue(ctx)
  mult_kernel = basic_kernels(ctx)

  GPUmeta(dev, ctx, queue, mult_kernel)
end

function basic_kernels(ctx)
  const mult_kernel = "
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
