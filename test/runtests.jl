using ImageQuilting
using GeoStatsBase
using GeoStatsImages
using Statistics
using Plots, VisualRegressionTests
using Test, Pkg, Random

# workaround GR warnings
ENV["GKSwstype"] = "100"

# environment settings
islinux = Sys.islinux()
istravis = "TRAVIS" ∈ keys(ENV)
datadir = joinpath(@__DIR__,"data")
visualtests = !istravis || (istravis && islinux)
if !istravis
  Pkg.add("Gtk")
  using Gtk
end

@testset "ImageQuilting.jl" begin
  @testset "Basic checks" begin
    # the output of a homogeneous image is also homogeneous
    TI = ones(20,20,20)
    reals = iqsim(TI, (10,10,10), size(TI))
    @test reals[1] == TI

    # categories are obtained from training image only
    ncateg = 3; TI = rand(0:ncateg, 20, 20, 20)
    reals = iqsim(TI, (10,10,10), size(TI))
    @test Set(reals[1]) ⊆ Set(TI)
  end

  @testset "Soft data" begin
    # trends with soft data
    TI = [zeros(10,20,1); ones(10,20,1)]
    trend = [zeros(20,10,1) ones(20,10,1)]
    reals = iqsim(TI, (10,10,1), size(TI), soft=[(trend,TI)], tol=1)
    @test mean(reals[1][:,1:10,:]) ≤ mean(reals[1][:,11:20,:])

    # no side effects with soft data
    TI = ones(20,20,20)
    TI[:,5,:] .= NaN
    aux = fill(1.0, size(TI))
    iqsim(TI, (10,10,10), size(TI), soft=[(aux,aux)])
    @test aux == fill(1.0, size(TI))

    # auxiliary variable with integer type
    TI = ones(20,20,20)
    aux = [i for i in 1:20, j in 1:20, k in 1:20]
    iqsim(TI, (10,10,10), size(TI), soft=[(aux,aux)])
    @test aux == [i for i in 1:20, j in 1:20, k in 1:20]
  end

  @testset "Hard data" begin
    # hard data is honored everywhere
    TI = ones(20,20,20)
    obs = rand(size(TI)...)
    data = Dict(CartesianIndex(i,j,k)=>obs[i,j,k] for i=1:20, j=1:20, k=1:20)
    reals = iqsim(TI, (10,10,10), size(TI), hard=data)
    @test reals[1] == obs

    # multiple realizations with hard data
    TI = ones(20,20,20)
    data = Dict(CartesianIndex(20,20,20)=>10)
    reals = iqsim(TI, (10,10,10), size(TI), hard=data, nreal=3)
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
    reals = iqsim(TI, (10,10,10), size(TI), hard=shape)
    @test all(isnan.(reals[1][.!active]))
    @test all(.!isnan.(reals[1][active]))

    # masked training image
    TI = ones(20,20,20)
    TI[:,5,:] .= NaN
    reals = iqsim(TI, (10,10,10), size(TI))
    @test reals[1] == fill(1.0, size(TI))
    TI[1,5,:] .= 0
    reals = iqsim(TI, (10,10,10), size(TI))
    @test reals[1] == fill(1.0, size(TI))

    # masked domain and masked training image
    TI = ones(20,20,20)
    TI[:,5,:] .= NaN
    aux = fill(1.0, size(TI))
    shape = Dict(CartesianIndex(i,j,k)=>NaN for i=1:20, j=5, k=1:20)
    reals = iqsim(TI, (10,10,10), size(TI), hard=shape)
    @test all(isnan.(reals[1][:,5,:]))
    @test all(reals[1][:,1:4,:] .== 1)
    @test all(reals[1][:,6:20,:] .== 1)
    reals = iqsim(TI, (10,10,10), size(TI), hard=shape, soft=[(aux,aux)])
    @test all(isnan.(reals[1][:,5,:]))
    @test all(reals[1][:,1:4,:] .== 1)
    @test all(reals[1][:,6:20,:] .== 1)
  end

  @testset "Minimum error cut" begin
    # 3D cut
    TI = ones(20,20,20)
    _, _, voxs = iqsim(TI, (10,10,10), overlap=(1/3,1/3,1/3), debug=true)
    @test 0 ≤ voxs[1] ≤ 1
  end

  @testset "Simulation paths" begin
    # different simulation paths
    for kind in [:raster,:dilation,:random]
      path = ImageQuilting.genpath((10,10,10), kind, Int[])
      @test length(path) == 1000
    end

    # data is visited first if present
    path = ImageQuilting.genpath((10,10,10), :data, [1,1000])
    @test path[1:2] == [1,1000] || path[1:2] == [1000,1]
  end

  @testset "Voxel reuse" begin
    # mean voxel reuse is in range [0,1]
    TI = rand(20,20,20)
    μ, σ = voxelreuse(TI, (10,10,10), nreal=1)
    @test 0 ≤ μ ≤ 1
  end

  if visualtests
    for TIname in ["Strebelle","StoneWall"]
      Random.seed!(2017)
      TI = training_image(TIname)[1:50,1:50,:]
      reals = iqsim(TI, (30,30,1), size(TI), nreal=4)
      ps = [heatmap(real[:,:,1]) for real in reals]
      @plottest plot(ps...) joinpath(datadir,"Reals"*TIname*".png") !istravis
    end
    for TIname in ["StoneWall","WalkerLake"]
      Random.seed!(2017)
      TI = training_image(TIname)[1:20,1:20,:]
      @plottest voxelreuseplot(TI) joinpath(datadir,"Voxel"*TIname*".png") !istravis
    end
  end

  @testset "GeoStats.jl API" begin
    geodata = PointSetData(Dict(:variable => [1.,0.,1.]), [25. 50. 75.; 25. 75. 50.])
    grid    = RegularGrid{Float64}(100,100)
    problem = SimulationProblem(geodata, grid, :variable, 3)

    TI = training_image("Strebelle")
    inactive = [CartesianIndex(i,j,1) for i in 1:30 for j in 1:30]
    solver = ImgQuilt(:variable => (TI=TI, tilesize=(30,30,1), inactive=inactive))

    Random.seed!(2017)
    solution = solve(problem, solver)
    @test keys(solution.realizations) ⊆ [:variable]

    incomplete_solver = ImgQuilt()
    @test_throws AssertionError solve(problem, incomplete_solver)

    if visualtests
      @plottest plot(solution,size=(1000,300)) joinpath(datadir,"GeoStatsAPI.png") !istravis
    end
  end

  # if ImageQuilting.cl ≠ nothing && ImageQuilting.clfft ≠ nothing
    # @testset "GPU support" begin
      # # CPU and GPU give same results
      # TI = ones(20,20,20)
      # TI[10:end,:,:] = 2
      # Random.seed!(0); realscpu = iqsim(TI, (10,10,10), size(TI), gpu=false)
      # Random.seed!(0); realsgpu = iqsim(TI, (10,10,10), size(TI), gpu=true)
      # @test realscpu[1] == realsgpu[1]
    # end
  # end
end
