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

# hard data is honored everywhere
TI = ones(20,20,20)
obs = rand(size(TI))
data = HardData([(i,j,k)=>obs[i,j,k] for i=1:20, j=1:20, k=1:20])
reals = iqsim(TI, 10, 10, 10, size(TI)..., hard=data)
@test all(reals[1] .== obs)

# irregular grids via hard data conditioning
TI = ones(20,20,20)
shape = HardData()
active = trues(TI)
for i=1:20, j=1:20, k=1:20
  if (i-10)^2 + (j-10)^2 + (k-10)^2 < 25
    push!(shape, (i,j,k)=>NaN)
    active[i,j,k] = false
  end
end
reals = iqsim(TI, 10, 10, 10, size(TI)..., hard=shape)
@test all(isnan(reals[1][!active]))
@test all(!isnan(reals[1][active]))
