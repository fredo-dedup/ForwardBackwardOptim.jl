# tests 

# data generation
srand(0)
Np = 50
θ0 = randn(Np)
θ0[ rand(Np) .> 0.1 ] = 0.

Nobs = 10000
x = randn(Np, Nobs)
u = vec(θ0' * x) + 0.1randn(Nobs)
y = 1 ./ (1. + exp(-u))
y = map(Float64, y .> 0.5 )


# convergence comparisons
model = m.riskmodel(m.LinearPred(Np), m.LogisticLoss())


θ1 = m.fbs(model, zeros(Np), x, y, 
           reg=m.SqrL2Reg(10.0), λ=2e-3, maxsteps=100, maxtime=10)
θ2 = m.fista(model, zeros(Np), x, y, 
             reg=m.SqrL2Reg(10.0), λ=2e-3, maxsteps=100, maxtime=10)
θ3 = m.fasta(model, zeros(Np), x, y, 
             reg=m.SqrL2Reg(10.0), λ=1e-3, maxsteps=100, maxtime=10)

@test sumabs2(θ1-θ2) < 1e-4
@test sumabs2(θ2-θ3) < 1e-4

θ1 = m.fbs(model, zeros(Np), x, y, 
           reg=m.L1Reg(10.0), λ=2e-3, maxsteps=1000, maxtime=10, cbinterval=1000)
θ2 = m.fista(model, zeros(Np), x, y, 
             reg=m.L1Reg(10.0), λ=2e-3, maxsteps=1000, maxtime=10, cbinterval=1000)
θ3 = m.fasta(model, zeros(Np), x, y, 
             reg=m.L1Reg(10.0), λ=1e-3, maxsteps=100, maxtime=10, cbinterval=100)

@test sumabs2(θ1-θ2) < 1e-4
@test sumabs2(θ2-θ3) < 1e-4


# fbs!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=SqrL2Reg(1.0), λ=1e-3, maxsteps=20, maxtime=10, cbinterval = 2)

# fista!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=SqrL2Reg(1.0), λ=1e-3, maxsteps=20, maxtime=10, cbinterval = 2)

# fasta!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=SqrL2Reg(1.0), λ=1e-3, maxsteps=20, maxtime=10, cbinterval = 1)
# fasta2!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=SqrL2Reg(1.0), λ=1e-3, maxsteps=20, maxtime=10, cbinterval = 1)



# fbs!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=L1Reg(10.0), λ=1e-3, maxsteps=20, maxtime=10, cbinterval = 2)

# fista!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=L1Reg(10.0), λ=1e-3, maxsteps=20, maxtime=10, cbinterval = 2)

# fasta!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=L1Reg(10.0), λ=1e-3, maxsteps=20, maxtime=10, cbinterval = 1)
# fasta2!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=L1Reg(10.0), λ=1e-3, maxsteps=20, maxtime=10, cbinterval = 1)

# fasta!(riskmodel(LinearPred(Np), MyLoss()), zeros(Np), x, y, 
#      reg=L1Reg(10.0), λ=10., maxsteps=50, maxtime=10, cbinterval = 2)
# fasta2!(riskmodel(LinearPred(Np), MyLoss()), ones(Np), x, y, 
#      reg=L1Reg(10.0), λ=10., maxsteps=50, maxtime=10, cbinterval = 2)
# fasta2!(riskmodel(LinearPred(Np), MyLoss()), ones(Np), x, y, 
#      reg=L1Reg(1.0), λ=10., maxsteps=50, maxtime=10, cbinterval = 2)


