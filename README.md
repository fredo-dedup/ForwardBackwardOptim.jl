# ForwardBackwardOptim

_Forward-backward splitting optimization algorithms_

|Julia release (0.3)  | Julia nightly (0.4)|
|---------------|:-----------:|
|[![ForwardBackwardOptim](http://pkg.julialang.org/badges/ForwardBackwardOptim_release.svg)](http://pkg.julialang.org/?pkg=ForwardBackwardOptim&ver=release)           |  [![ForwardBackwardOptim](http://pkg.julialang.org/badges/ForwardBackwardOptim_nightly.svg)](http://pkg.julialang.org/?pkg=ForwardBackwardOptim&ver=nightly) |

Latest (Julia nightly & release) [![Build Status](https://travis-ci.org/fredo-dedup/ForwardBackwardOptim.jl.svg?branch=master)](https://travis-ci.org/fredo-dedup/ForwardBackwardOptim.jl)  

[![Coverage Status](https://coveralls.io/repos/fredo-dedup/ForwardBackwardOptim.jl/badge.png?branch=master)](https://coveralls.io/r/fredo-dedup/ForwardBackwardOptim.jl?branch=master)


This package provides 3 functions to minimize a function under a regularization constraint:
    - `fbs()` for the standard forward-backward splitting with a constant, user-supplied, step size
    - `fista()` for the accelerated forward-backward splitting (with a constant, user-supplied, step size)
    - `fasta()` for the accelerated forward-backward splitting with adaptative step size and backtracking

The function and regularization definitions are provided by the package `EmpiricalRisks`, see [here](https://github.com/lindahua/EmpiricalRisks.jl).

Install with : `Pkg.add("ReverseDiffSource")`

Example:
```julia
using ForwardBackwardOptim

# Generate data for a logistic regression
Np = 50
θ0 = randn(Np)
θ0[ rand(Np) .> 0.1 ] = 0.

Nobs = 10000
x = randn(Np, Nobs)
u = vec(θ0' * x) + 0.1randn(Nobs)
y = 1 ./ (1. + exp(-u))
y = float64( y .> 0.5 )

# Define model (see EmpiricalRisks.jl)
model = riskmodel(LinearPred(Np), LogisticLoss())

# Launch optimization with L2 regularization
θ2 = fasta(model, 
          zeros(Np), x, y, 
          reg      = SqrL2Reg(10.0), # regularizer definition
          maxsteps = 100,            # maximum number of steps
          maxtime  = 10)             # maximum runtime 10 sec

# with L1 regularization
θ1 = fasta(model, 
          zeros(Np), x, y, 
          reg      = L1Reg(10.0), # regularizer definition
          maxsteps = 100,         # maximum number of steps
          maxtime  = 10)          # maximum runtime 10 sec
```