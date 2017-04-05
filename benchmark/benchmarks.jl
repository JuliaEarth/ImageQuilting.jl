using ImageQuilting
using GeoStatsImages
using PkgBenchmark

srand(2017)

TI = rand(50, 50, 50)
iqsim(TI, 10, 10, 10, size(TI)...) # warm up

@benchgroup "2D simulation" ["iqsim"] begin
  for TIname in ["Strebelle", "StoneWall"]
    TI = training_image(TIname)
    println(size(TI))
    flush(STDOUT)
    @bench TIname iqsim(TI, 30, 30, 1, size(TI)..., nreal=3)
  end
end

@benchgroup "3D simulation" ["iqsim"] begin
  TI = training_image("StanfordV")
  @bench "StanfordV" iqsim(TI, 30, 30, 10, size(TI)...)
end
