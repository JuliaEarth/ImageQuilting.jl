using ImageQuilting
using Base.Test

# the output of a homogeneous image is also homogeneous
@test all(iqsim(ones(100,100,100), 20, 10, 5, 100, 100, 100)[1] .== 1)
