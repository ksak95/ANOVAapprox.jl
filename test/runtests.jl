using ANOVAapprox
using Test
using Random
using Aqua

Aqua.test_all(ANOVAapprox, ambiguities = false)
Aqua.test_ambiguities(ANOVAapprox)

include("TestFunctionPeriodic.jl")
include("TestFunctionCheb.jl")

using .TestFunctionPeriodic
using .TestFunctionCheb

rng = MersenneTwister(1234)

tests =
    ["misc", "cheb_fista", "cheb_lsqr", "per_lsqr", "per_fista", "wav_lsqr", "mixed_lsqr"]

for t in tests
    include("$(t).jl")
end
