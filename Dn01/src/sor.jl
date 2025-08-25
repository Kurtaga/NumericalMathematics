function sor(A::SparseMatrix, b::Vector{Float64}, ω::Float64)
    n, _ = size(A)
    x0 = zeros(n)
    x, iters = solveSor(A, b, x0, ω)
    return x, iters
end

function solveSor(A::SparseMatrix, b::Vector{Float64}, 
             x0::Vector{Float64}, ω::Float64; tol=1e-10, maxiter=10_000)

    n, _ = size(A)
    x = copy(x0)

    for k in 1:maxiter
        for i in 1:n
            # compute sum over row i, skipping diagonal
            σ = 0.0
            aii = 0.0
            for (val, j) in zip(A.V[i], A.I[i])
                if j == i
                    aii = val
                else
                    σ += val * x[j]
                end
            end

            # SOR update
            if aii == 0.0
                error("Zero diagonal element at row $i")
            end

            newxi = (1-ω)*x[i] + (ω/aii) * (b[i] - σ)
            x[i] = newxi
        end

        # check residual ∞-norm
        r = A * x - b
        if norm(r, Inf) < tol # biggest element in residual
            return x, k
        end
    end

    error("SOR did not converge within $maxiter iterations")
end
