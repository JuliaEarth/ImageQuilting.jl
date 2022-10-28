# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

include("imfilter.cpu.jl")
include("imfilter.cuda.jl")
include("imfilter.opencl.jl")

function select_default_kernel()
  if CUDA.functional()
    CUDAKernel()
  elseif !isempty(cl.platforms()) && !isempty(cl.devices())
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