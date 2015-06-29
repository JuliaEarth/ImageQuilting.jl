using ImageQuilting
using Base.Test

# the output of a homogeneous image is also homogeneous
@test all(iqsim(ones(20,20,20), 10, 10, 10, 20, 20, 20)[1] .== 1)
