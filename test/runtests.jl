using Base.Test

reload("ForwardBackwardOptim")
m = ForwardBackwardOptim

tests = [
    "optims"
]

for t in tests
    tfile = string(t, ".jl")
    println("  * $tfile ...")
    include(tfile)
end

println("Finished testing.")