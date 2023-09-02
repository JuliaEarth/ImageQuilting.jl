using ImageQuilting
using Meshes
using GeoTables
using GeoStatsBase
using GeoStatsImages
using ImageFiltering
using LinearAlgebra
using Statistics
using CUDA
using Test, Random

# list of tests
testfiles = ["lowapi.jl", "highapi.jl"]

@testset "ImageQuilting.jl" begin
  for testfile in testfiles
    println("Testing $testfile...")
    include(testfile)
  end
end
