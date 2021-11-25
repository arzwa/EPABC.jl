using SmoothTree, NewickTree
using SmoothTree: MSC, CCD, randsplits

struct MSCModel{M}
    model::M
end

function (model::MSCModel)(θ::Number)
    SmoothTree.setdistance!(model.model.tree, exp.(θ))
    return model
end

function (model::MSCModel)(θ::Vector)
    SmoothTree.setdistance_internal!(model.model.tree, exp.(θ))
    return model
end

getmodel(model::MSCModel, ccd::CCD) = MSCModel(model.model(ccd))
Base.rand(model::MSCModel) = randsplits(model.model)

# simulate data
# univariate
S = nw"((smo,(((gge,iov),(xtz,dzq)),sgt)),jvs);"
SmoothTree.setdistance!(S, exp(0.75)) 
dmodel = MSC(S)
data = [CCD(dmodel, randsplits(dmodel)) for i=1:200]

μ = 0.
v = 1.
M = 50000
acceptfun = (x,y)->log(rand()) < logpdf(x,y)
model = MSCModel(dmodel)
alg = GaussianEPABC(data, model, acceptfun, μ, v, M)

trace = ep_pass!(alg)
trace = [trace; ep_pass!(alg)]
#trace = [trace; ep_pass!(alg)]

mtrace = permutedims(mapreduce(x->x.μ, hcat, trace))
p1 = plot(mtrace)
hline!([0.75], ls=:dash, color=:black)
p2 = plot(Normal(trace[end].μ, √trace[end].v))
vline!([0.75])
plot(p1, p2)

# multivariate
m = mtrace[end,:]
v = sqrt.(diag(trace[end].Σ))
p = plot(grid=false, legend=false)
for i=1:5
    plot!(Normal(m[i], v[i]))
end
plot(p)





