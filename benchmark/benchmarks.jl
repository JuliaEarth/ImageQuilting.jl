using ImageQuilting
using GeoStatsImages
using PkgBenchmark

srand(2017)

@benchgroup "2D simulation" begin
  Herten = training_image("Herten")
  Strebelle = training_image("Strebelle")
  @bench "Herten" iqsim(Herten, 20, 20, 1, 100, 100, 1, nreal=3)
  @bench "Strebelle" iqsim(Strebelle, 20, 20, 1, 200, 200, 1, nreal=3)
end

@benchgroup "3D simulation" begin
  TI = training_image("Flumy")
  nx, ny, nz = size(TI)

  shape = HardData()
  for idx in find(isnan.(TI))
      i,j,k = ind2sub(size(TI), idx)
      push!(shape, (i,j,k)=>NaN)
  end

  AUX = [i for i in 1:nx, j in 1:ny, k in 1:nz]
  sdata = SoftData(AUX, _ -> AUX)

  @bench "Flumy" iqsim(TI, 50, 50, 20, size(TI)..., hard=shape, soft=sdata, tol=.01)
end
