# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function init_imfilter_kernel(::CPUMethod) end

const array_kernel(::CPUMethod, array) = array

const view_kernel(::CPUMethod, array, I) = view(array, I)

function imfilter_kernel(::CPUMethod, img, krn)
  imfilter_cpu(img, krn)
end

function imfilter_cpu(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end
