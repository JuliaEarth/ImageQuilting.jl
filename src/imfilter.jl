# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_cpu(img, kern)
  imfilter(img, centered(kern), Inner(), Algorithm.FFT())
end
