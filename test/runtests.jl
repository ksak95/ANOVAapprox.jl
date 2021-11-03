using ANOVAapprox
using Test
using Random
using Aqua

Aqua.test_all(ANOVAapprox)

include("TestFunctionPeriodic.jl")
include("TestFunctionCheb.jl")

using .TestFunctionPeriodic
using .TestFunctionCheb

rng = MersenneTwister(1234)

tests = ["cheb_fista", "cheb_lsqr", "per_lsqr", "per_fista"]

for t in tests
    include("$(t).jl")
end
