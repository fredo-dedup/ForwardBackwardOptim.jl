################################################################################
#
#   fbs : Standard Forward Backward splitting
#
################################################################################

function fbs!{T<:FloatingPoint}(
    rmodel::SupervisedRiskModel,     # model definition
    θ::StridedArray{T},              # initial parameter guess
    x::StridedArray{T},              # inputs
    y::StridedArray{T};              # responses
    reg::Regularizer = ZeroReg(),    # regularizer
    λ::T             = one(T),       # step size
    cbinterval::Int  = 0,            # callback interval
    callback         = fbs_trace,    # callback function
    maxsteps::Int    = 100,          # max number of steps 
    maxtime::Int     = 0         )   # max calculation time (0 = no limit)

    g    = similar(θ)
    θbar = similar(θ)
    starttime = time()
    for t in 1:maxsteps
        if (maxtime != 0) && (time() - starttime > maxtime)
            break
        end

        copy!(θbar, θ)
        v_risk, _ = value_and_addgrad!(rmodel, 0., g, 1., θbar, x, y)
        axpy!(-λ, g, θbar)
        prox!(reg, θ, θbar, λ)

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            callback(t, θ, v_risk, value(reg, θ))
        end
    end

    θ
end

# Non mutating version
function fbs{T<:FloatingPoint}(
    rmodel::SupervisedRiskModel,     # model definition
    θ::StridedArray{T},              # initial parameter guess
    x::StridedArray{T},              # inputs
    y::StridedArray{T};              # responses
    vargs... )

    fbs!(rmodel, copy(θ), x, y; vargs...)
end