#### PERIODIC TEST SOLVER LSQR ####

using ANOVAapprox
using Test

include("TestFunctionPeriodic.jl")

d = 10
ds = 3
M = 10_000
max_iter = 50
bw = [100, 10, 6]
λs = [0.0, 1.0]


X = rand(d, M) .- 0.5
y = [TestFunctionPeriodic.f(X[:, i]) for i = 1:M]
X_test = rand(d, M) .- 0.5
y_test = [TestFunctionPeriodic.f(X_test[:, i]) for i = 1:M]

####  ####
#for i=1:200

ads = ANOVAapprox.approx(X, complex(y), ds, bw, "per")
@time ANOVAapprox.approximate(ads, lambda = λs)

#println("AR: ", sum(ANOVAapprox.get_AttributeRanking(ads, 0.0)))
#@test abs(sum(ANOVAapprox.get_AttributeRanking(ads, 0.0)) - 1) < 0.0001

#bw = ANOVAapprox.get_orderDependentBW(TestFunctionPeriodic.AS, [128, 32])

#aU = ANOVAapprox.approx(X, complex(y), TestFunctionPeriodic.AS, bw, "per")
#ANOVAapprox.approximate(aU, lambda = λs)

#end