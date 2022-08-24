using ImageQuilting
using Meshes
using GeoStatsBase
using GeoStatsImages
using ImageFiltering
using Statistics
using CUDA
using Plots; gr(size=(600,400))
using MeshPlots # TODO: replace by MeshViz
using ReferenceTests, ImageIO
using Test, Random

# workaround GR warnings
ENV["GKSwstype"] = "100"

# environment settings
isCI = "CI" ∈ keys(ENV)
islinux = Sys.islinux()
visualtests = !isCI || (isCI && islinux)
datadir = joinpath(@__DIR__,"data")

# list of tests
testfiles = [
  "lowapi.jl",
  "highapi.jl"
]

@testset "ImageQuilting.jl" begin
  for testfile in testfiles
    println("Testing $testfile...")
    include(testfile)
  end
end