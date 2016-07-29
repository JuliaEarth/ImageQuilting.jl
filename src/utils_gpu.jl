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

immutable GPUmeta
  dev
  ctx
  queue
  mult_kernel
end

function gpu_setup()
  @assert cl ≠ nothing "OpenCL.jl not installed, cannot use GPU"
  gpus = cl.devices(:gpu)
  @assert !isempty(gpus) "GPU not found, make sure drivers are installed"

  gpu = []
  gpunames = map(d -> d[:platform][:name], gpus)
  for vendor in ["NVIDIA","AMD","Intel"], (idx,name) in enumerate(gpunames)
    if contains(name, vendor)
      gpu = gpus[idx]
      break
    end
  end

  info("using GPU $(gpu[:name])")

  ctx = cl.Context(gpu)
  queue = cl.CmdQueue(ctx)
  mult_kernel = basic_kernels(ctx)

  GPUmeta(gpu, ctx, queue, mult_kernel)
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
  v = clfft.version()

  # clFFT releases support powers of 2, 3, 5, ...
  radices = [2,3,5]
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
