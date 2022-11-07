# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

include("kernel/imfilter_cpu.jl")
include("kernel/imfilter_cuda.jl")
include("kernel/imfilter_opencl.jl")

function opencl_functional()
  try
    !isempty(cl.devices())
  catch err
    if err isa cl.CLError
      false
    else
      rethrow(err)
    end
  end
end

function select_kernel_method()
  if CUDA.functional()
    CUDAMethod()
  elseif opencl_functional()
    OpenCLMethod()
  else
    CPUMethod()
  end
end

const kernel_method = select_kernel_method()

const init_imfilter_kernel() = init_imfilter_kernel(kernel_method)
const array_kernel(array) = array_kernel(kernel_method, array)
const view_kernel(array, I) = view_kernel(kernel_method, array, I)
const imfilter_kernel(img, krn) = imfilter_kernel(kernel_method, img, krn)