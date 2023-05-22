@testset "Inactive locations" begin
  sdata = georef((facies=[1.0, 0.0, 1.0],), [25.0 50.0 75.0; 25.0 75.0 50.0])
  sdomain = CartesianGrid(100, 100)
  problem = SimulationProblem(sdata, sdomain, :facies, 3)

  rng = MersenneTwister(2017)
  trainimg = geostatsimage("Strebelle")
  inactive = [CartesianIndex(i, j) for i in 1:30 for j in 1:30]
  solver = IQ(:facies => (trainimg=trainimg, tilesize=(30, 30), inactive=inactive), rng=rng)

  solution = solve(problem, solver)
  @test length(solution) == 3
  @test size(domain(solution[1])) == (100, 100)

  if visualtests
    @test_reference "data/GeoStatsAPI-1.png" plot(solution, size=(900, 300))
  end
end

@testset "Forward model" begin
  truthimg = geostatsimage("WalkerLakeTruth")
  trainimg = geostatsimage("WalkerLake")

  # forward model (blur filter)
  function forward(data)
    img = asarray(data, :Z)
    krn = KernelFactors.IIRGaussian([10, 10])
    fwd = imfilter(img, krn)
    georef((fwd=fwd,), domain(data))
  end

  # apply forward model to both images
  data = forward(truthimg)
  dataTI = forward(trainimg)

  problem = SimulationProblem(domain(truthimg), :Z => Float64, 3)

  rng = MersenneTwister(2017)
  solver = IQ(:Z => (trainimg=trainimg, tilesize=(27, 27), soft=(data, dataTI)), rng=rng)

  solution = solve(problem, solver)

  if visualtests
    @test_reference "data/GeoStatsAPI-2.png" plot(solution, size=(900, 300))
  end
end
