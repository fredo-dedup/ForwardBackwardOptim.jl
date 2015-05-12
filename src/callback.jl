# Callbacks functions

function fbs_trace(t, θ, loss, reg, λ)
    @printf("iter %5d : loss = %8.4f, reg = %8.4f, total = %8.4f, ",
            t, loss, reg, loss+reg)
    @printf("non zero = %d\n",
            λ, sum(θ .!= 0.))
end

function fista_trace(t, θ, loss, reg, λ, α)
    @printf("iter %5d : loss = %8.4f, reg = %8.4f, total = %8.4f, ", 
            t, loss, reg, loss+reg)
    @printf("α = %4.2f, non zero = %d\n", 
            λ, α, sum(θ .!= 0.))
end

function fasta_trace(t, θ, loss, reg, λ, α, bt)
    @printf("iter %5d : loss = %8.4f, reg = %8.4f, total = %8.4f, ", 
            t, loss, reg, loss+reg)
    @printf("λ = %8.5f, α = %4.2f, backtracks = %2d, non zero = %d\n", 
            λ, α, bt, sum(θ .!= 0.))
end

