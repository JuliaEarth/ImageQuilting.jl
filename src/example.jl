using Images

include("imquilt.jl")

function example()
    srand(2014) # make sure results are reproducible
    imquilt(imread(joinpath("images","radishes.jpg")), 64, 5 , show=true)
    imquilt(imread(joinpath("images","weave.jpg"))   , 80, 5 , show=true)
    imquilt(imread(joinpath("images","brick.jpg"))   , 80, 5 , show=true)
    imquilt(imread(joinpath("images","apples.gif"))  , 31, 15, show=true)
    imquilt(imread(joinpath("images","btile.tif"))   , 16, 15, show=true)
end
