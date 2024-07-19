@doc raw"""
    get_l2error( a::approx, λ::Float64 )::Float64

This function computes the relative ``\ell_2`` error on the training nodes for regularization parameter `λ`.
"""
function get_l2error(a::approx, λ::Float64)::Float64
    y_eval = evaluate(a, λ)
    return norm(y_eval - a.y) / norm(a.y)
end

@doc raw"""
    get_l2error( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, λ::Float64 )::Float64

This function computes the relative ``\ell_2`` error on the data `X` and `y` for regularization parameter `λ`.
"""
function get_l2error(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
)::Float64
    y_eval = evaluate(a, X, λ)
    return norm(y_eval - y) / norm(y)
end

@doc raw"""
    get_l2error( a::approx )::Dict{Float64,Float64}

This function computes the relative ``\ell_2`` error on the training nodes for all regularization parameters.
"""
function get_l2error(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_l2error(a, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    get_l2error( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, )::Dict{Float64,Float64}

This function computes the relative ``\ell_2`` error on the data `X` and `y` for all regularization parameters.
"""
function get_l2error(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
)::Dict{Float64,Float64}
    return Dict(λ => get_l2error(a, X, y, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    get_mse( a::approx, λ::Float64 )::Float64

This function computes the mean square error (mse) on the training nodes for regularization parameter `λ`.
"""
function get_mse(a::approx, λ::Float64)::Float64
    y_eval = evaluate(a, λ)
    return 1 / length(a.y) * (norm(y_eval - a.y)^2)
end

@doc raw"""
    get_mse( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, λ::Float64 )::Float64

This function computes the mean square error (mse) on the data `X` and `y` for regularization parameter `λ`.
"""
function get_mse(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
)::Float64
    y_eval = evaluate(a, X, λ)
    return 1 / length(y) * (norm(y_eval - y)^2)
end

@doc raw"""
    get_mse( a::approx )::Dict{Float64,Float64}

This function computes the mean square error (mse) on the training nodes for all regularization parameters.
"""
function get_mse(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_mse(a, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    get_mse( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, )::Dict{Float64,Float64}

This function computes the mean square error (mse) on the data `X` and `y` for all regularization parameters.
"""
function get_mse(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
)::Dict{Float64,Float64}
    return Dict(λ => get_mse(a, X, y, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    get_mad( a::approx, λ::Float64 )::Float64

This function computes the mean absolute deviation (mad) on the training nodes for regularization parameter `λ`.
"""
function get_mad(a::approx, λ::Float64)::Float64
    y_eval = evaluate(a, λ)
    return 1 / length(a.y) * norm(y_eval - a.y, 1)
end

@doc raw"""
    get_mad( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, λ::Float64 )::Float64

This function computes the mean absolute deviation (mad) on the data `X` and `y` for regularization parameter `λ`.
"""
function get_mad(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
)::Float64
    y_eval = evaluate(a, X, λ)
    return 1 / length(y) * norm(y_eval - y, 1)
end

@doc raw"""
    get_mad( a::approx )::Dict{Float64,Float64}

This function computes the mean absolute deviation (mad) on the training nodes for all regularization parameters.
"""
function get_mad(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_mad(a, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    get_mse( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, )::Dict{Float64,Float64}

This function computes the mean absolute deviation (mad) on the data `X` and `y` for all regularization parameters.
"""
function get_mad(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
)::Dict{Float64,Float64}
    return Dict(λ => get_mad(a, X, y, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    get_L2error( a::approx, norm::Float64, bc_fun::Function, λ::Float64 )::Float64

This function computes the relative ``L_2`` error of the function given the norm `norm` and a function that returns the basis coefficients `bc_fun` for regularization parameter `λ`.
"""
function get_L2error(a::approx, norm::Float64, bc_fun::Function, λ::Float64)::Float64
    if a.basis == "per" || a.basis == "cos" || a.basis == "cheb" || a.basis == "std"
        error = norm^2
        index_set = get_IndexSet(a.trafo.setting, size(a.X, 1))

        for i = 1:size(index_set, 2)
            k = index_set[:, i]
            error += abs(bc_fun(k) - a.fc[λ][i])^2 - abs(bc_fun(k))^2
        end

        return sqrt(error) / norm
    else
        error("The L2-error is not implemented for this basis")
    end
end

@doc raw"""
    get_L2error( a::approx, norm::Float64, bc_fun::Function )::Dict{Float64,Float64}

This function computes the relative ``L_2`` error of the function given the norm `norm` and a function that returns the basis coefficients `bc_fun` for all regularization parameters.
"""
function get_L2error(a::approx, norm::Float64, bc_fun::Function)::Dict{Float64,Float64}
    return Dict(λ => get_L2error(a, norm, bc_fun, λ) for λ in collect(keys(a.fc)))
end

################

function get_acc(a::approx, λ::Float64)::Float64
    return count(sign.(evaluate(a, λ)) .== a.y)/length(a.y)*100.00
end

function get_acc(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
)::Float64
    return count(sign.(evaluate(a, X, λ)) .== y)/length(y)*100.0
end

function get_acc(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_acc(a, λ) for λ in collect(keys(a.fc)))
end

function get_acc(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
)::Dict{Float64,Float64}
    return Dict(λ => get_acc(a, X, y, λ) for λ in collect(keys(a.fc)))
end
#=
function get_svn(a::approx, λ::Float64)::Float64
    y_eval = evaluate(a, λ)
    return count((a.y .* y_eval) .< 1.0)
end

function get_svn(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_svn(a, λ) for λ in collect(keys(a.fc)))
end
=#

function get_auc(a::approx, λ::Float64)::Float64
    y_eval = evaluate(a, λ)
    y_sc = (y_eval .- minimum(y_eval)) / (maximum(y_eval) - minimum(y_eval))
    y = a.y
    y[y .== -1.0] .= 0
    y[y .== 1.0] .= 1
    y_int = Vector{Int64}(y)
    return MultivariateAnomalies.auc(y_sc, y_int)
end

function get_auc(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
)::Float64
    y_eval = evaluate(a, X, λ)
    y_sc = (y_eval .- minimum(y_eval)) / (maximum(y_eval) - minimum(y_eval))
    y_[y .== -1.0] .= 0
    y_[y .== 1.0] .= 1
    y_int = Vector{Int64}(y_)
    return MultivariateAnomalies.auc(y_sc, y_int)
end

function get_auc(a::approx)::Dict{Float64,Float64}
    return Dict(λ => get_auc(a, λ) for λ in collect(keys(a.fc)))
end

function get_auc(
    a::approx,
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
)::Dict{Float64,Float64}
    return Dict(λ => get_auc(a, X, y, λ) for λ in collect(keys(a.fc)))
end