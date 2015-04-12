using ImageQuilting
using Base.Test

# the output of a homogeneous image is also homogeneous
@test all(iqsim(ones(50,50,50), 10, 10, 10, 50, 50, 50)[1] .== 1)
