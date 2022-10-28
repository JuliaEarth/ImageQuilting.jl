# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

# using the CPU kernel
const array_kernel(array, ::CPUKernel) = array
const view_kernel(array, I, ::CPUKernel) = view(array, I)

function imfilter_kernel(img, krn, ::CPUKernel)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end
