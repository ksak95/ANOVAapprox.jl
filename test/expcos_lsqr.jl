using Test
using ANOVAapprox

d = 6
ds = 2
M = 10_000
max_iter = 50
bw = [100, 10]
λs = [0.0, 1.0]

dcos = [false,false,false,false,false,false]

X = rand(d, M)./ 2

y = [TestFunctionPeriodic.f(X[:, i]) for i = 1:M]
X_test = rand(d, M)./ 2
y_test = [TestFunctionPeriodic.f(X_test[:, i]) for i = 1:M]

####  ####

ads = ANOVAapprox.approx(X, complex(y), ds, bw, "expcos", dcos)
ANOVAapprox.approximate(ads, lambda = λs)

println("AR: ", sum(ANOVAapprox.get_AttributeRanking(ads, 0.0)))
@test abs(sum(ANOVAapprox.get_AttributeRanking(ads, 0.0)) - 1) < 0.0001

bw = ANOVAapprox.get_orderDependentBW(TestFunctionPeriodic.AS, [128, 32])

aU = ANOVAapprox.approx(X, complex(y), TestFunctionPeriodic.AS, bw, "expcos", dcos)
ANOVAapprox.approximate(aU, lambda = λs)

err_L2_ds =
    ANOVAapprox.get_L2error(ads, TestFunctionPeriodic.norm(), TestFunctionPeriodic.fc)[0.0]
err_L2_U =
    ANOVAapprox.get_L2error(aU, TestFunctionPeriodic.norm(), TestFunctionPeriodic.fc)[0.0]
err_l2_ds = ANOVAapprox.get_l2error(ads)[0.0]
err_l2_U = ANOVAapprox.get_l2error(aU)[0.0]
err_l2_rand_ds = ANOVAapprox.get_l2error(ads, X_test, complex(y_test))[0.0]
err_l2_rand_U = ANOVAapprox.get_l2error(aU, X_test, complex(y_test))[0.0]

LibTest.plotGSIS(ads, 0.0)
LibTest.plotGSIS(aU, 0.0)

println("== PERIODIC LSQR ==")
println("L2 ds: ", err_L2_ds)
println("L2 U: ", err_L2_U)
println("l2 ds: ", err_l2_ds)
println("l2 U: ", err_l2_U)
println("l2 rand ds: ", err_l2_rand_ds)
println("l2 rand U: ", err_l2_rand_U)

@test err_L2_ds < 0.01
@test err_L2_U < 0.005
@test err_l2_ds < 0.01
@test err_l2_U < 0.005
@test err_l2_rand_ds < 0.01
@test err_l2_rand_U < 0.005
