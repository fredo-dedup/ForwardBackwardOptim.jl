################################################################################
#
#   fasta : Accelerated Forward Backward splitting with 
#            adaptative step size and backtracking
#
################################################################################

const NLookback=10 # previous steps lookback to decide backtracking

function fasta!{T<:FloatingPoint}(
    rmodel::SupervisedRiskModel,     # model definition
    θ::StridedArray{T},              # initial parameter guess
    x::StridedArray{T},              # inputs
    y::StridedArray{T};              # responses
    reg::Regularizer = ZeroReg(),    # regularizer
    λ::T             = one(T),       # initial step size
    cbinterval::Int  = 0,            # callback interval
    callback         = fasta_trace,  # callback function
    maxsteps::Int    = 100,          # max number of steps 
    maxtime::Int     = 0         )   # max calculation time (0 = no limit)

    g              = similar(θ) # holds gradient
    prevg          = similar(θ) # gradient of previous step
    deltag         = similar(θ) # g - predg

    prevθ          = similar(θ) # θ of previous step
    deltaθ         = similar(θ) # θ - prevθ

    dualθ          = copy(θ)    # dual θ
    candidatedualθ = similar(θ) # candidate dual θ
    deltadualθ     = similar(θ)

    vhist     = fill(-Inf, NLookback) # past values

    function fbstep!(θ1, g, θ0, λ)
        copy!(θ1, θ0)
        axpy!(-λ, g, θ1)
        prox!(reg, θ1, θ1, λ)
    end

    ### Loop initialization
    starttime = time()
    α, μ      = 1., 0.      # acceleration parameters
    v, _      = value_and_addgrad!(rmodel, 0., g, 1., dualθ, x, y)
    vhist[1]  = v

    cbinterval > 0 && callback(0, θ, v, value(reg, θ), λ, α, 0)
    for t in 1:maxsteps
        (maxtime != 0) && (time() - starttime > maxtime) && break

        # acceleration
        if t >= 2
            if dot(g, deltadualθ) >= 0.
                α = 1.
                μ = 0.
            else
                candα = (1. + sqrt(1+4α*α))/2.
                μ = (α - 1.) / candα
                α = candα
            end
        end

        # adaptative step size
        if t >= 2
            copy!(deltag, g) ; axpy!(-1., prevg, deltag)
            tmp = dot(deltadualθ, deltag)
            λs  = dot(deltadualθ, deltadualθ) / tmp
            λm  = tmp / dot(deltag, deltag)
            if λs > 0. && λm > 0.
                λ = (λm / λs > 0.5) ? λm : λs - 0.5λm
            end
        end

        # update, with backtracking
        btsteps = 0
        maxv = maximum(vhist)
        copy!(prevθ, θ)
        while btsteps < 10
            fbstep!(θ, g, dualθ, λ)
            copy!(deltaθ, θ) ; axpy!(-1., prevθ, deltaθ)
            copy!(candidatedualθ, θ)
            axpy!(μ, deltaθ, candidatedualθ)

            candidatev = value(rmodel, candidatedualθ, x, y)
            deltadualθ = candidatedualθ - dualθ # TODO : preallocate
            if candidatev < maxv + dot(deltadualθ, g) + sumabs2(deltadualθ)/2λ
                break
            end
            λ /= 2.
            btsteps += 1
        end
        copy!(dualθ     , candidatedualθ)

        # update variables for next step
        copy!(prevg     , g)
        v, _ = value_and_addgrad!(rmodel, 0., g, 1., dualθ, x, y)
        unshift!(vhist, v)
        pop!(vhist)

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            callback(t, θ, v, value(reg, θ), λ, α, btsteps)
        end
    end

    θ
end

# Non mutating version
function fasta{T<:FloatingPoint}(
    rmodel::SupervisedRiskModel,     # model definition
    θ::StridedArray{T},              # initial parameter guess
    x::StridedArray{T},              # inputs
    y::StridedArray{T};              # responses
    vargs... )

    fasta!(rmodel, copy(θ), x, y; vargs...)
end
