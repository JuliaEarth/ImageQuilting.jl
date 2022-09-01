using BenchmarkTools

const TOLERANCE = 1e-3

function allclose(x::AbstractArray{T}, y::AbstractArray{T}; rtol=1e-5, atol=1e-8) where {T}
    @assert length(x) == length(y)
    @inbounds begin
        for i in 1:length(x)
            xx, yy = x[i], y[i]
            if !(isapprox(xx, yy; rtol=rtol, atol=atol))
                return false
            end
        end
    end
    return true
end

function test_imfilter(N)

    @info "start"

    img = rand(Float64,(N,N,N))
    krn = rand(Float64,(10,10,10))

    X = Int(round(N/2))

    @info "FIRST RUN - GPU - CUDA"
    r1 = @btime imfilter_cuda(CuArray($img), $krn)
    @info "check = ", r1[X,X,X]

    @info("SECOND RUN - GPU - OpenCL")
    global GPU = gpu_setup()
    r2 = @btime imfilter_opencl($img, $krn)
    @info "check = ", r2[X,X,X]

    @info "THIRD RUN -  CPU - ImageFiltering"
    r3 = @btime imfilter($img, centered($krn), Inner(), Algorithm.FFT())
    @info "check = ", r3[X,X,X]

    @info "r1 & r2", allclose(r1, r2; rtol=1e-2, atol=1e-3)
    @info "r1 & r3", allclose(r1, r3; rtol=1e-2, atol=1e-3)
    @info "r2 & r3", allclose(r2, r3; rtol=1e-2, atol=1e-3)

    @info "end"

end