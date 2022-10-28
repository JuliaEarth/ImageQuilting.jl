# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

include("kernel/imfilter_cpu.jl")
include("kernel/imfilter_cuda.jl")
include("kernel/imfilter_opencl.jl")

function has_opencl_available()
  try
    hasdevices = !isempty(OpenCL.cl.devices())
    return hasdevices
  catch err
    if err isa cl.CLError
      return false
    else
      rethrow(err)
    end
  end
end

function select_default_kernel()
  if CUDA.functional()
    CUDAKernel()
  elseif has_opencl_available()
    OpenCLKernel()
  else
    CPUKernel()
  end
end

const default_kernel = select_default_kernel()

# define functions with default kernel
const array_kernel(array) = array_kernel(array, default_kernel)
const view_kernel(array, I) = view_kernel(array, I, default_kernel)
const imfilter_kernel(img, krn) = imfilter_kernel(img, krn, default_kernel)