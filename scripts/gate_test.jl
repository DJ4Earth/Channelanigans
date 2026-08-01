ctl = open(ENV["PERF_CTL_FIFO"], read=true, write=true)
ack = open(ENV["PERF_ACK_FIFO"], read=true, write=true)

A = rand(1000, 1000); B = rand(1000, 1000)
println(sum(A * B))                                   # warmup — must NOT be counted

println(ctl, "enable");  flush(ctl); readline(ack)
C = A * B                                             # the counted region
println(ctl, "disable"); flush(ctl); readline(ack)

println(sum(C))

t0 = time_ns()
println(ctl, "enable");  flush(ctl); readline(ack)
println(ctl, "disable"); flush(ctl); readline(ack)
println("control round trip: ", (time_ns() - t0) / 1e6, " ms")