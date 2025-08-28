function curve_length(traj)
    L = 0.0
    for k in 2:length(traj)
        L += norm3(traj[k] - traj[k-1])
    end
    return L
end