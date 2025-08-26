function cross3(a::Vector{Float64}, b::Vector{Float64})
    return [
        a[2]*b[3] - a[3]*b[2],
        a[3]*b[1] - a[1]*b[3],
        a[1]*b[2] - a[2]*b[1]
    ]
end

function norm3(v::Vector{Float64})
    return sqrt(v[1]^2 + v[2]^2 + v[3]^2)
end