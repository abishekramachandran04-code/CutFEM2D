function [F_D, F_L, C_D, C_L] = calc_aero_forces(mesh, conn, edof, dof, Re, dt, U, U_n, f_source, obstacle_nodes, D, U_inf)
    [I, J, V, F_raw] = navierstokes(mesh, conn, edof, dof, Re, dt, U, U_n, f_source);
    K = sparse(I, J, V, dof.ndof, dof.ndof);
    K = K / dt;
    F_raw = F_raw / dt;
    U_cpu = gather(U);
    Reaction_Forces = K * U_cpu - F_raw;

    F_D = 0;  F_L = 0;
    for i = 1:length(obstacle_nodes)
        node = obstacle_nodes(i);
        F_D = F_D - Reaction_Forces(dof.node(node, 1));
        F_L = F_L - Reaction_Forces(dof.node(node, 2));
    end
    q_inf = 0.5 * 1.0 * U_inf^2;
    C_D = F_D / (q_inf * D);
    C_L = F_L / (q_inf * D);
end