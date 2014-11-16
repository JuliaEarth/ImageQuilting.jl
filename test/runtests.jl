using ImageQuilting
using Images
using Base.Test

# the output of a homogeneous image is also homogeneous
@test all(imquilt(grayim(ones(100,100)), 12, 5) .== 1)
