# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

function imfilter_resource(resource, img, kern)
  imfilter(img, centered(kern), Inner(), Algorithm.FFT())
end
