module ANOVAapprox

using GroupedTransforms,
 LinearAlgebra, IterativeSolvers, LinearMaps, Distributed, SpecialFunctions, Optim

bases = ["per", "cos", "cheb", "std", "chui1", "chui2", "chui3", "chui4", "mixed"]
types = Dict(
    "per" => ComplexF64,
    "cos" => Float64,
    "cheb" => Float64,
    "std" => Float64,
    "chui1" => Float64,
    "chui2" => Float64,
    "chui3" => Float64,
    "chui4" => Float64,
    "mixed" => ComplexF64,
)
vtypes = Dict(
    "per" => Vector{ComplexF64},
    "cos" => Vector{Float64},
    "cheb" => Vector{Float64},
    "std" => Vector{Float64},
    "chui1" => Vector{Float64},
    "chui2" => Vector{Float64},
    "chui3" => Vector{Float64},
    "chui4" => Vector{Float64},
    "mixed" => Vector{ComplexF64},
)
gt_systems = Dict(
    "per" => "exp",
    "cos" => "cos",
    "cheb" => "cos",
    "std" => "cos",
    "chui1" => "chui1",
    "chui2" => "chui2",
    "chui3" => "chui3",
    "chui4" => "chui4",
    "mixed" => "mixed",
)

function get_orderDependentBW(U::Vector{Vector{Int}}, N::Vector{Int})::Vector{Int}
    N_bw = zeros(Int64, length(U))

    for i = 1:length(U)
        if U[i] == []
            N_bw[i] = 0
        else
            N_bw[i] = N[length(U[i])]
        end
    end

    return N_bw
end

function bisection(l, r, fun; maxiter = 1_000)
    lval = fun(l)
    rval = fun(r)

    sign(lval)*sign(rval) == 1 && error("bisection: root is not between l and r")
    if lval > 0
        gun = fun
        fun = t -> -gun(t)
    end

    m = 0.0
    for _ in 1:maxiter
        m = (l+r)/2
        mval = fun(m)
        abs(mval) < 1e-16 && break
        if mval < 0
            l = m
            lval = mval
        else
            r = m
            rval = mval
        end
    end
    return m
end


"""
    C = fitrate(X, y)
fits a function of the form
  ``(C[4] - x)*(C[1] + C[2]*x^C[3])``

# Input
 - `X::Vector{Float64}`
 - `y::Vector{Float64}`
 - `verbose::Bool = false`

# Output
 - `C::Vector{Float64}`: coefficients of the approximation
"""
function fitrate(X, y::Vector{Float64}; verbose::Bool = false, )::Vector{Float64}
    # no rate
    length(unique(y)) == 1 && return [0.0, 0.0, -100.0, length(X)+1]
    
    # delete zeros at the end
    idx = length(y) - findfirst(reverse(y) .!= 0) + 1
    X = X[1:idx]
    y = y[1:idx]

    function f(C::Vector)
        return norm(log.((maximum(X)+exp(C[4]) .- X) .* (exp(C[1]) .+ exp(C[2])*X.^(C[3]))) - log.(y), 1)
    end

    x0 = [log(y[argmax(X)]), log((y[argmin(X)]-y[argmax(X)])*minimum(X)^3), -3.0, 1]
    if verbose
        @show res = optimize(f, x0)
    else
        res = optimize(f, x0)
    end

    C = Optim.minimizer(res)
    C[1] = exp.(C[1])
    C[2] = exp.(C[2])
    C[4] = exp(C[4])+maximum(X)

    C[3] >= 0 && return [0.0, 0.0, -100.0, length(X)+1]

    return C
end

function testrate(S::Vector{Vector{Float64}},C::Vector{Vector{Float64}},t::Float64)::Vector{Bool}
    E = [((C[i][4]).-(1:length(S[i]))).*(C[i][1].+C[i][2].*(1:length(S[i])).^(C[i][3])) for i=1:length(C)]
    return [sum(abs.(E[i].-S[i]))/length(S[i])<t for i=1:length(C)]
end


include("fista.jl")
include("approx.jl")
include("errors.jl")
include("analysis.jl")

end # module
