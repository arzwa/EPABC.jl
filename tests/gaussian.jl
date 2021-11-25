using Pkg; Pkg.activate(@__DIR__)
using EPABC, Distributions, LinearAlgebra
using EPABC: GaussianEPABC, ep_pass!, ep_iteration!, l2norm

# the model object
struct MvnModel{T,V}
    μ::T
    Σ::V
end
(m::MvnModel)(θ) = MvnModel(θ, m.Σ)
Base.rand(m::MvnModel) = rand(MvNormal(m.μ, m.Σ))

# simulate data from a MvNormal
n = 200
true_μ = randn(3)
true_Σ = diagm(ones(3))
model  = MvnModel(true_μ, true_Σ)
data   = [rand(model) for i=1:n]

# set up EP-ABC 
# μ and Σ are the prior for the approximation
# Note: we assume known variance in the model, so the only unknown
# parameters are the means.
μ = zeros(3) 
Σ = diagm(ones(3))
M = 20000
ϵ = 1.5
alg = GaussianEPABC(data, model, l2norm, μ, Σ, ϵ, M)

# do three passes
res = ep_pass!(alg)
res = [res ; ep_pass!(alg)]
res = [res ; ep_pass!(alg)]

# some plots
using Plots, StatsPlots
default(grid=false, legend=false, guidefont=9)
p1 = plot(hcat(first.(res)...)')
hline!(true_μ, color=:black, ls=:dot, xlabel="iteration")
m = res[end][1]  
S = res[end][2]
p2 = plot(xlabel="mean")
for i=1:3
    plot!(Normal(m[i], sqrt(S[i,i])), fill=true, fillalpha=0.1)
end
vline!(p2, true_μ, color=:black, ls=:dot)
plot(p1, p2, size=(650,300))

# compare with analytic posterior
function uvposterior(μ₀, V₀, V, n, x̄)
    ϕ = (V/n + V₀) 
    postmean = (V₀/ϕ)*x̄ + (V/ϕ)*μ₀
    postvar = (1/V₀ + n/V)^(-1)
    return Normal(postmean, √postvar)
end

x̄ = mean(first.(data))
post = uvposterior(0, 1, true_Σ[1], n, x̄) 
plot(post, color=:black, ls=:dash, fill=true, fillalpha=0.1)
plot!(Normal(m[1], √S[1]))
