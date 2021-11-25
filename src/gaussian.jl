# Gaussian EP-ABC, general implementation
# Arthur Zwaenepoel 2021

# l2norm function
function l2norm(x, y)
    z = x - y
    return sqrt(z' * z)
end

# transformation from natural to moment parameters for a MVN
function natural2moment_gaussian(r, Q)
    Qinv = Q^(-1)
    Σ = -0.5*Qinv
    (μ=-0.5*Qinv*r, Σ=Symmetric(Σ))
end

# transformation from moment to natural parameters for a MVN
function moment2natural_gaussian(μ, Σ)
    Σinv = Σ^(-1)
    (r=Σinv*μ, Q=-0.5*Σinv)
end

"""
    GaussianEPABC(...)
    GaussianEPABC(data, model, norm, μ, Σ, ϵ, M)

Fairly general implementation for approximating a target distribution
using EP-ABC with a multivariate Gaussian as approximating
distribution. See the `gaussian.jl` file for an example.
"""
mutable struct GaussianEPABC{T,V,X,M}
    data  ::X
    model ::M          # sampling model
    accept::Function   # ABC distance
    r     ::T          # prior/current r
    Q     ::V          # prior/current Q
    rs    ::Vector{T}  # site r's
    Qs    ::Vector{V}  # site Q's
    Z     ::Float64    # marginal likelihood
    M     ::Int        # number of simulations/iteration
end

# initialize a Gaussian EP-ABC object based on a moment
# parameterization.
function GaussianEPABC(data, model, accept, μ, Σ, M)
    n = length(data)
    r, Q = moment2natural_gaussian(μ, Σ)
    rs = [zero(r) for i=1:n]
    Qs = [zero(Q) for i=1:n]
    GaussianEPABC(data, model, accept, r, Q, rs, Qs, NaN, M)
end

# do a full EP-ABC pass over the data
ep_pass!(alg) = map(i->ep_iteration!(alg, i), 1:length(alg.data))

# default getmodel fallback
getmodel(model, datapoint) = model

# a single EP-ABC iteration (i.e. for a single data point/block)
function ep_iteration!(alg::GaussianEPABC, i)
    @unpack data, model, accept, r, Q, rs, Qs, M = alg
    # first we get the "cavity"
    r -= rs[i]  # note that this does not change alg.r (I believe)
    Q -= Qs[i]
    μ, Σ = natural2moment_gaussian(r, Q)
    # simulate from the cavity
    if !isposdef(Σ)
        @warn "not posdef?" Σ
        return (μ=μ, Σ=Σ, Z=alg.Z)
    end
    θ = rand(MvNormal(μ, Σ), M)
    # this extra step allows to tailor the model to the data point if
    # necessary
    model_i = getmodel(model, data[i])
    # simulate from the hybrid
    sims = map(1:M) do m
        ysim = rand(model_i(θ[:,m]))
        accept(data[i], ysim)
    end
    # compute empirical moments
    acc, μ_, Σ_ = 0, zero(μ), zero(Σ)
    for (m, d) in enumerate(sims)
        !d && continue
        acc += 1
        μ_ += θ[:,m]
        Σ_ += θ[:,m]*θ[:,m]'
    end
    @info "$(acc/M) accepted"
    # update site parameters by moment matching
    # the if is to prevent posdef exceptions, should be based on M
    if acc/M > 0.001   # XXX is this a good value? it appears to work
        Z = acc/M
        μ = μ_/acc
        Σ = Σ_/acc - μ*μ'
        Σinv = Σ^(-1)
        alg.Z = Z
        alg.r .= Σinv*μ
        alg.Q .= -0.5*Σinv
        alg.rs[i] = alg.r - r
        alg.Qs[i] = alg.Q - Q
    end
    return (μ=μ, Σ=Σ, Z=alg.Z)
end

# univariate implementation
function ep_iteration!(alg::GaussianEPABC{T}, i) where T<:Number
    @unpack data, model, accept, r, Q, rs, Qs, M = alg
    r -= rs[i]
    Q -= Qs[i]
    μ = -r/(2Q)
    v = -1/(2Q)
    θ = rand(Normal(μ, √v), M)
    model_i = getmodel(model, data[i])
    sims = map(1:M) do m
        ysim = rand(model_i(θ[m]))
        accept(data[i], ysim)
    end
    acc, μ_, v_ = 0, zero(μ), zero(v)
    for (m, d) in enumerate(sims)
        !d && continue
        acc += 1
        μ_ += θ[m]
        v_ += θ[m]^2
    end
    @info "Z=$(acc/M), μ=$(μ_/acc)"
    # update site parameters by moment matching
    # the if is to prevent posdef exceptions, should be based on M
    if acc/M > 0.001   # XXX is this a good value?
        Z = acc/M
        μ = μ_/acc
        v = v_/acc - μ^2
        alg.Z = Z
        alg.r = μ/v
        alg.Q = -1/(2v)
        alg.rs[i] = alg.r - r
        alg.Qs[i] = alg.Q - Q
    end
    return (μ=μ, v=v, Z=alg.Z)
end


