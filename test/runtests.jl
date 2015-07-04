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

# trends with soft data
TI = [zeros(10,20,1); ones(10,20,1)]
trend = [zeros(20,10,1) ones(20,10,1)]
reals = iqsim(TI, 10, 10, 1, size(TI)..., soft=SoftData(trend, x -> x), cutoff=1)
@test mean(reals[1][:,1:10,:]) ≤ mean(reals[1][:,11:20,:])
