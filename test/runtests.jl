using ImageQuilting
using Meshes
using GeoStatsBase
using GeoStatsImages
using ComputationalResources
  using Statistics
using Plots; gr(size=(600,400))
using ReferenceTests, ImageIO
using Test, Random

# workaround GR warnings
ENV["GKSwstype"] = "100"

# environment settings
isCI = "CI" ∈ keys(ENV)
islinux = Sys.islinux()
visualtests = !isCI || (isCI && islinux)
datadir = joinpath(@__DIR__,"data")
testresource = CPU1()

@testset "ImageQuilting.jl" begin
  @testset "Basic checks" begin
    # the output of a homogeneous image is also homogeneous
    TI = ones(20,20,20)
    reals = iqsim(TI, (10,10,10), size(TI), resource=testresource)
    @test reals[1] == TI

    # categories are obtained from training image only
    ncateg = 3; TI = rand(0:ncateg, 20, 20, 20)
    reals = iqsim(TI, (10,10,10), size(TI), resource=testresource)
    @test Set(reals[1]) ⊆ Set(TI)
  end

  @testset "Soft data" begin
    # trends with soft data
    TI = [zeros(10,20,1); ones(10,20,1)]
    trend = [zeros(20,10,1) ones(20,10,1)]
    reals = iqsim(TI, (10,10,1), size(TI), soft=[(trend,TI)], tol=1, resource=testresource)
    @test mean(reals[1][:,1:10,:]) ≤ mean(reals[1][:,11:20,:])

    # no side effects with soft data
    TI = ones(20,20,20)
    TI[:,5,:] .= NaN
    aux = fill(1.0, size(TI))
    iqsim(TI, (10,10,10), size(TI), soft=[(aux,aux)], resource=testresource)
    @test aux == fill(1.0, size(TI))

    # auxiliary variable with integer type
    TI = ones(20,20,20)
    aux = [i for i in 1:20, j in 1:20, k in 1:20]
    iqsim(TI, (10,10,10), size(TI), soft=[(aux,aux)], resource=testresource)
    @test aux == [i for i in 1:20, j in 1:20, k in 1:20]
  end

  @testset "Hard data" begin
    # hard data is honored everywhere
    TI = ones(20,20,20)
    obs = rand(size(TI)...)
    data = Dict(CartesianIndex(i,j,k)=>obs[i,j,k] for i=1:20, j=1:20, k=1:20)
    reals = iqsim(TI, (10,10,10), size(TI), hard=data, resource=testresource)
    @test reals[1] == obs

    # multiple realizations with hard data
    TI = ones(20,20,20)
    data = Dict(CartesianIndex(20,20,20)=>10)
    reals = iqsim(TI, (10,10,10), size(TI), hard=data, nreal=3, resource=testresource)
    for real in reals
      @test real[20,20,20] == 10
    end
  end

  @testset "Masked grids" begin
    # masked simulation domain
    TI = ones(20,20,20)
    shape = Dict{CartesianIndex{3},Real}()
    active = trues(size(TI))
    for i=1:20, j=1:20, k=1:20
      if (i-10)^2 + (j-10)^2 + (k-10)^2 < 25
        push!(shape, CartesianIndex(i,j,k)=>NaN)
        active[i,j,k] = false
      end
    end
    reals = iqsim(TI, (10,10,10), size(TI), hard=shape, resource=testresource)
    @test all(isnan.(reals[1][.!active]))
    @test all(.!isnan.(reals[1][active]))

    # masked training image
    TI = ones(20,20,20)
    TI[:,5,:] .= NaN
    reals = iqsim(TI, (10,10,10), size(TI), resource=testresource)
    @test reals[1] == fill(1.0, size(TI))
    TI[1,5,:] .= 0
    reals = iqsim(TI, (10,10,10), size(TI), resource=testresource)
    @test reals[1] == fill(1.0, size(TI))

    # masked domain and masked training image
    TI = ones(20,20,20)
    TI[:,5,:] .= NaN
    aux = fill(1.0, size(TI))
    shape = Dict(CartesianIndex(i,j,k)=>NaN for i=1:20, j=5, k=1:20)
    reals = iqsim(TI, (10,10,10), size(TI), hard=shape, resource=testresource)
    @test all(isnan.(reals[1][:,5,:]))
    @test all(reals[1][:,1:4,:] .== 1)
    @test all(reals[1][:,6:20,:] .== 1)
    reals = iqsim(TI, (10,10,10), size(TI), hard=shape, soft=[(aux,aux)], resource=testresource)
    @test all(isnan.(reals[1][:,5,:]))
    @test all(reals[1][:,1:4,:] .== 1)
    @test all(reals[1][:,6:20,:] .== 1)
  end

  @testset "Minimum error cut" begin
    # 3D cut
    TI = ones(20,20,20)
    _, _, voxs = iqsim(TI, (10,10,10), overlap=(1/3,1/3,1/3), debug=true, resource=testresource)
    @test 0 ≤ voxs[1] ≤ 1

    A = ones(20,20)
    B = ones(20,20)
    C = ImageQuilting.boykov_kolmogorov_cut(A, B, 1)
    @test all(C[1:end-1,:] .== 1)
    @test all(C[end,:] .== 0)
    C = ImageQuilting.boykov_kolmogorov_cut(A, B, 2)
    @test all(C[:,1:end-1] .== 1)
    @test all(C[:,end] .== 0)
  end

  @testset "Simulation paths" begin
    # different simulation paths
    for kind in [:raster,:dilation,:random]
      rng = MersenneTwister(123)
      path = ImageQuilting.genpath(rng, (10,10,10), kind, Int[])
      @test length(path) == 1000
    end

    # data is visited first if present
    rng = MersenneTwister(123)
    path = ImageQuilting.genpath(rng, (10,10,10), :data, [1,1000])
    @test path[1:2] == [1,1000] || path[1:2] == [1000,1]
  end

  @testset "Voxel reuse" begin
    # mean voxel reuse is in range [0,1]
    TI = rand(20,20,20)
    μ, σ = voxelreuse(TI, (10,10,10), nreal=1)
    @test 0 ≤ μ ≤ 1
  end

  if visualtests
    for (TIname,var) in [("Strebelle",:facies),("StoneWall",:Z)]
      rng = MersenneTwister(2017)
      img = geostatsimage(TIname)
      TI = asarray(img, var)[1:50,1:50,:]
      reals = iqsim(TI, (30,30,1), size(TI), nreal=4, rng=rng, resource=testresource)
      ps = [heatmap(real[:,:,1]) for real in reals]
      @test_reference "data/Reals$(TIname).png" plot(ps...)
    end
    for TIname in ["StoneWall","WalkerLake"]
      rng = MersenneTwister(2017)
      img = geostatsimage(TIname)
      TI = asarray(img, :Z)[1:20,1:20,:]
      @test_reference "data/Voxel$(TIname).png" voxelreuseplot(TI, rng=rng)
    end
  end

  @testset "GeoStats.jl API" begin
    sdata   = georef((facies=[1.,0.,1.],), [25. 50. 75.; 25. 75. 50.])
    sdomain = CartesianGrid(100,100)
    problem = SimulationProblem(sdata, sdomain, :facies, 3)

    rng = MersenneTwister(2017)
    trainimg = geostatsimage("Strebelle")
    inactive = [CartesianIndex(i,j) for i in 1:30 for j in 1:30]
    solver = IQ(:facies => (trainimg=trainimg, tilesize=(30,30), inactive=inactive), rng=rng, resource = testresource)

    solution = solve(problem, solver)
    @test length(solution) == 3
    @test size(domain(solution[1])) == (100,100)

    incomplete_solver = IQ()
    @test_throws ErrorException solve(problem, incomplete_solver)

    if visualtests
      @test_reference "data/GeoStatsAPI.png" plot(solution,size=(900,300))
    end
  end
end
