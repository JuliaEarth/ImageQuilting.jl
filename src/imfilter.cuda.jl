# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

@platform aware function init_imfilter_kernel({accelerator_count::(@atleast 1), accelerator_api::CUDA_API})
  println("Running on CUDA GPU")
end

@platform aware function array_kernel({accelerator_count::(@atleast 1), accelerator_api::CUDA_API}, array) CuArray(array) end

@platform aware function view_kernel({accelerator_count::(@atleast 1), accelerator_api::CUDA_API}, array, I) Array(array[I]) end

@platform aware function imfilter_kernel({accelerator_count::(@atleast 1), accelerator_api::CUDA_API}, img, krn)
    imfilter_cuda(img,krn)
 end


function imfilter_cuda(img, krn)
 
   # pad kernel to common size with image
   padkrn = CUDA.zeros(size(img))
   copyto!(padkrn, CartesianIndices(krn), CuArray(krn), CartesianIndices(krn))
 
   # perform ifft(fft(img) .* conj.(fft(krn)))
   fftimg = img |> CUFFT.fft
   fftkrn = padkrn |> CuArray |> CUFFT.fft
   result = (fftimg .* conj.(fftkrn)) |> CUFFT.ifft
 
   # recover result
   finalsize = size(img) .- (size(krn) .- 1)
   real.(result[CartesianIndices(finalsize)]) |> Array

end

