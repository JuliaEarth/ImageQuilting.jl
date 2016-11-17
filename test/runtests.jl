using ImageQuilting
using Base.Test

@testset "Basic checks" begin
  # the output of a homogeneous image is also homogeneous
  TI = ones(20,20,20)
  reals = iqsim(TI, 10, 10, 10, size(TI)...)
  @test reals[1] == TI

  # categories are obtained from training image only
  ncateg = 3; TI = rand(RandomDevice(), 0:ncateg, 20, 20, 20)
  reals = iqsim(TI, 10, 10, 10, size(TI)..., categorical=true)
  @test Set(reals[1]) ⊆ Set(TI)
end

@testset "Soft data" begin
  # trends with soft data
  TI = [zeros(10,20,1); ones(10,20,1)]
  trend = [zeros(20,10,1) ones(20,10,1)]
  reals = iqsim(TI, 10, 10, 1, size(TI)..., soft=SoftData(trend, x -> x), tol=1)
  @test mean(reals[1][:,1:10,:]) ≤ mean(reals[1][:,11:20,:])

  # no side effects with soft data
  TI = ones(20,20,20)
  TI[:,5,:] = NaN
  aux = ones(TI)
  trend = SoftData(aux, x -> aux)
  iqsim(TI, 10, 10, 10, size(TI)..., soft=trend)
  @test aux == ones(TI)
end

@testset "Hard data" begin
  # hard data is honored everywhere
  TI = ones(20,20,20)
  obs = rand(size(TI))
  data = HardData((i,j,k)=>obs[i,j,k] for i=1:20, j=1:20, k=1:20)
  reals = iqsim(TI, 10, 10, 10, size(TI)..., hard=data)
  @test reals[1] == obs

  # multiple realizations with hard data
  TI = ones(20,20,20)
  data = HardData((20,20,20)=>10)
  reals = iqsim(TI, 10, 10, 10, size(TI)..., hard=data, nreal=3)
  for real in reals
    @test real[20,20,20] == 10
  end
end

@testset "Masked grids" begin
  # masked simulation domain
  TI = ones(20,20,20)
  shape = HardData()
  active = trues(TI)
  for i=1:20, j=1:20, k=1:20
    if (i-10)^2 + (j-10)^2 + (k-10)^2 < 25
      push!(shape, (i,j,k)=>NaN)
      active[i,j,k] = false
    end
  end
  reals = iqsim(TI, 10, 10, 10, size(TI)..., hard=shape)
  @test all(isnan(reals[1][!active]))
  @test all(!isnan(reals[1][active]))

  # masked training image
  TI = ones(20,20,20)
  TI[:,5,:] = NaN
  reals = iqsim(TI, 10, 10, 10, size(TI)...)
  @test reals[1] == ones(TI)
  TI[1,5] = 0
  reals = iqsim(TI, 10, 10, 10, size(TI)..., categorical=true)
  @test reals[1] == ones(TI)

  # masked domain and masked training image
  TI = ones(20,20,20)
  TI[:,5,:] = NaN
  trend = SoftData(ones(TI), x -> ones(TI))
  shape = HardData((i,j,k)=>NaN for i=1:20, j=5, k=1:20)
  reals = iqsim(TI, 10, 10, 10, size(TI)..., hard=shape)
  @test all(isnan(reals[1][:,5,:]))
  @test all(reals[1][:,1:4,:] .== 1)
  @test all(reals[1][:,6:20,:] .== 1)
  reals = iqsim(TI, 10, 10, 10, size(TI)..., hard=shape, soft=trend)
  @test all(isnan(reals[1][:,5,:]))
  @test all(reals[1][:,1:4,:] .== 1)
  @test all(reals[1][:,6:20,:] .== 1)
end

@testset "Minimum error cut" begin
  # 3D cut
  TI = ones(20,20,20)
  for cut in [:dijkstra,:boykov]
    _, _, voxs = iqsim(TI, 10, 10, 10, size(TI)..., overlapx=1/3, overlapy=1/3, overlapz=1/3, cut=cut, debug=true)
    @test 0 ≤ voxs[1] ≤ 1
  end
end

@testset "Simulation paths" begin
  # different simulation paths
  for kind in [:rasterup,:rasterdown,:dilation,:random]
    path = ImageQuilting.genpath((10,10,10), kind)
    @test length(path) == 1000
  end

  # datum is visited first if present
  path = ImageQuilting.genpath((10,10,10), :datum, [(1,1,1),(10,10,10)])
  @test path[1:2] == [1,1000] || path[1:2] == [1000,1]
end

@testset "Voxel reuse" begin
  # mean voxel reuse is in range [0,1]
  TI = rand(20,20,20)
  μ, σ = voxelreuse(TI, 10, 10, 10, nreal=1)
  @test 0 ≤ μ ≤ 1
end
