using ImageQuilting
using Images
using Base.Test

# the output of a homogeneous image is also homogeneous
@test all(imquilt(grayim(ones(100,100)), 12, 5) .== 1)

# tile size must fit in image
@test_throws ArgumentError imquilt(grayim(eye(10)),20,5)

# tile size must be at least 12
@test_throws ArgumentError imquilt(grayim(eye(20)),10,5)
