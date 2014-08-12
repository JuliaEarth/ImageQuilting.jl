include("imquilt.jl")

using Images
using ImageQuilting

srand(2014) # make sure results are reproducible

imquilt(imread(joinpath("images","radishes.jpg")), 64, 5)
imquilt(imread(joinpath("images","weave.jpg")), 80, 5)
imquilt(imread(joinpath("images","brick.jpg")), 80, 5)
imquilt(imread(joinpath("images","apples.gif")), 31, 15)
imquilt(imread(joinpath("images","btile.tif")), 16, 15)
