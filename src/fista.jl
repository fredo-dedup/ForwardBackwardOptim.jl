################################################################################
#
#   fista : Accelerated Forward Backward splitting
#
################################################################################

function fista!{T<:FloatingPoint}(
    rmodel::SupervisedRiskModel,     # model definition (prediction model and loss function)
    θ::StridedArray{T},              # initial parameter guess
    x::StridedArray{T},              # inputs
    y::StridedArray{T};              # responses
    reg::Regularizer = ZeroReg(),    # regularizer
    λ::T             = one(T),       # step size
    cbinterval::Int  = 0,            # callback interval
    callback         = fista_trace,    # callback function
    maxsteps::Int    = 100,          # max number of steps 
    maxtime::Int     = 0         )   # max calculation time (0 = no limit)

    g      = similar(θ)  # holds gradient
    dualθ  = copy(θ)     # dual θ
    prevθ  = copy(θ)     # θ of previous step
    deltaθ = similar(θ)  # variation of theta

    function fbstep!(θ1, g, θ0, λ)
        copy!(θ1, θ0)
        axpy!(-λ, g, θ1)
        prox!(reg, θ1, θ1, λ)
    end

    α    = 1.      # acceleration parameter
    starttime = time()
    for t in 1:maxsteps
        if (maxtime != 0) && (time() - starttime > maxtime)
            break
        end

        v_risk, _ = value_and_addgrad!(rmodel, 0., g, 1., dualθ, x, y)
        fbstep!(θ, g, dualθ, λ)
        deltaθ = θ - prevθ
        if dot(g, deltaθ) >= 0.
            α = 1.
            μ = 0.
        else
            prevα = α
            α = 1. + sqrt(1+4α*α)/2.
            μ = (prevα - 1.) / α
        end
        copy!(dualθ, θ)
        axpy!(μ, deltaθ, dualθ)
        copy!(prevθ, θ)

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            callback(t, θ, v_risk, value(reg, θ), λ, α)
        end
    end

    θ
end

# Non mutating version
function fista{T<:FloatingPoint}(
    rmodel::SupervisedRiskModel,     # model definition
    θ::StridedArray{T},              # initial parameter guess
    x::StridedArray{T},              # inputs
    y::StridedArray{T};              # responses
    vargs... )

    fista!(rmodel, copy(θ), x, y; vargs...)
end
