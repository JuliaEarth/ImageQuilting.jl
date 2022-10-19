# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

# OpenCL program kernels

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
  prog = OpenCL.cl.Program(ctx, source=mult_kernel) |> OpenCL.cl.build!
  OpenCL.cl.Kernel(prog, "mult")
end

function build_conj_kernel(ctx)
  conj_kernel = "
  __kernel void conj(__global double2 *a)
  {
    int gid = get_global_id(0);
    a[gid].y = -a[gid].y;
  }
  "
  prog = OpenCL.cl.Program(ctx, source=conj_kernel) |> OpenCL.cl.build!
  conj_kernel = OpenCL.cl.Kernel(prog, "conj")
end
