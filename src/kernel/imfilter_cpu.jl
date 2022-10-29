# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

const array_kernel(array, ::CPUMethod) = array

const view_kernel(array, I, ::CPUMethod) = view(array, I)

function imfilter_kernel(img, krn, ::CPUMethod)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end
