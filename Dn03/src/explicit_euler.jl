
# ODE right-hand side
function rhs(u, ∇F1, ∇F2)
    x, y, z = u
    g1 = ∇F1(x, y, z)
    g2 = ∇F2(x, y, z)
    return cross3(g1, g2)
end


# explicit Euler step
function explicit_euler_step(u, h, ∇F1, ∇F2)
    v = rhs(u, ∇F1, ∇F2)
    return u .+ h * (v / norm3(v))
end

# integrate
function integrate_explicit_euler(u0, h, ∇F1, ∇F2; max_steps=250000, min_steps=100)
    tol = 3h
    u = u0
    traj = [u0]
    for k in 1:max_steps
        u = explicit_euler_step(u, h, ∇F1, ∇F2)
        push!(traj, u)
        if k > min_steps && norm3(u - u0) < tol
            println("Closed loop detected at step $k")
            break
        end
    end
    return traj
end
