using Dn01
using Graphs 
using GraphRecipes, Plots

G = circularLadder(8)
t = range(0, 2pi, 9)[1:end-1]
x = cos.(t)
y = sin.(t)
točke = hcat(hcat(x, y)', zeros(2, 8))
fix = 1:8

embed!(G, fix, točke)

println("Coordinates of all vertices:")
println(točke)

graphplot(G, 
          x = točke[1, :], 
          y = točke[2, :], 
          curves = false, 
            markersize = 0.21)


            m, n = 6, 6
G = Graphs.grid((m, n), periodic=false)
# vogali imajo stopnjo 2
vogali = filter(v -> degree(G, v) <= 2, vertices(G))
točke = zeros(2, n * m)
točke[:, vogali] = [0 0 1 1; 0 1 0 1]
embed!(G, vogali, točke)
graphplot(G, x=točke[1, :], y=točke[2, :], curves=false)
