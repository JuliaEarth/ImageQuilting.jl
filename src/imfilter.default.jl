# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

@platform default function init_imfilter_kernel()
  println("Running on DEFAULT PLATFORM")
end

@platform default function array_kernel(array) array end

@platform default function view_kernel(array, I) view(array, I) end

@platform default function imfilter_kernel(img, krn)
  imfilter(img, centered(krn), Inner(), Algorithm.FFT())
end


