using ImageQuilting
using Base.Test

# the output of a homogeneous image is also homogeneous
TI = ones(20,20,20)
reals = iqsim(TI, 10, 10, 10, size(TI)...)
@test all(reals[1] .== 1)

# categories are obtained from training image only
ncateg = 3; TI = rand(RandomDevice(), 0:ncateg, 20, 20, 20)
reals = iqsim(TI, 10, 10, 10, size(TI)..., categorical=true)
@test Set(reals[1]) ⊆ Set(TI)
