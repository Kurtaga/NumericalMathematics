
struct SparseMatrix
    V::Vector{Vector{Float64}}   # values per row
    I::Vector{Vector{Int}}       # column indices per row
    n::Int                       # dimension (square matrix) TODO: check if i need this
end

Base.size(A::SparseMatrix) = (A.n, A.n)

function Base.getindex(A::SparseMatrix, i::Int, j::Int)
    for (k, col) in enumerate(A.I[i])
        if col == j
            return A.V[i][k]
        end
    end
    return 0.0
end

function Base.setindex!(A::SparseMatrix, val::Float64, i::Int, j::Int)
    for (k, col) in enumerate(A.I[i])
        if col == j
            A.V[i][k] = val
            return
        end
    end
    # if not found, add new entry
    push!(A.I[i], j)
    push!(A.V[i], val)
end

function Base.:*(A::SparseMatrix, x::Vector{Float64})
    n, _ = size(A)
    y = zeros(n)
    for i in 1:n
        for (val, j) in zip(A.V[i], A.I[i])
            y[i] += val * x[j]
        end
    end
    return y
end

function Base.:-(A::SparseMatrix)
    Vnew = [(-).(row) for row in A.V]   # negate all values
    Inew = [copy(row) for row in A.I]   # column indices unchanged
    return SparseMatrix(Vnew, Inew, A.n)
end

function sparseFromDense(M::Matrix{Float64})
    n = size(M, 1)
    V = Vector{Vector{Float64}}(undef, n)
    I = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        V[i] = Float64[]
        I[i] = Int[]
        for j in 1:n
            if M[i,j] != 0
                push!(V[i], M[i,j])
                push!(I[i], j)
            end
        end
    end
    return SparseMatrix(V, I, n)
end