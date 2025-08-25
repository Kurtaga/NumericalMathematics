function sparseMatrix(G::AbstractGraph, sprem)
    v_to_i = Dict([sprem[i] => i for i in eachindex(sprem)])
    m = length(sprem)

    M = zeros(m, m)
    for i = 1:m
        v = sprem[i]
        nodes = neighbors(G, v)
        for v2 in nodes
            if haskey(v_to_i, v2)
                j = v_to_i[v2]
                M[i, j] = 1
            end
        end
        M[i,i] = -length(nodes)
    end
    return sparseFromDense(M)  # convert to your SparseMatrix type
end

function rhsVector(G::AbstractGraph, sprem, coords)
    set = Set(sprem)
    m = length(sprem)
    b = zeros(m)
    for i in 1:m
        v = sprem[i]
        for v2 in neighbors(G, v)
            if !(v2 in set) # only fixed
                b[i] -= coords[v2]
            end
        end
    end
    return b
end

function embed!(G::AbstractGraph, fix, points; ω = 1.0)
    sprem = setdiff(vertices(G), fix)   # free vertices
    dim, _ = size(points)               # 2D or 3D
    A = sparseMatrix(G, sprem)

    iters_all = Int[]
    for k = 1:dim
        b = rhsVector(G, sprem, points[k, :])
        x, iters = sor(-A, -b, ω)
        push!(iters_all, iters)
        points[k, sprem] = x
    end

    return maximum(iters_all)
end


function circularLadder(n)
    G = SimpleGraph(2 * n)
    # prvi cikel
    for i = 1:n-1
        add_edge!(G, i, i + 1)
    end
    add_edge!(G, 1, n)
    # drugi cikel
    for i = n+1:2n-1
        add_edge!(G, i, i + 1)
    end
    add_edge!(G, n + 1, 2n)
    # povezave med obema cikloma
    for i = 1:n
        add_edge!(G, i, i + n)
    end
    return G
end