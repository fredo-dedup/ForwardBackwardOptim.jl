module ForwardBackwardOptim

using Compat
using Reexport
@reexport using EmpiricalRisks

import Base.LinAlg: axpy!

# source files

include("callback.jl")
include("fbs.jl")
include("fista.jl")
include("fasta.jl")

export  fbs!, fbs,
        fista!, fista,
        fasta!, fasta

end # module
