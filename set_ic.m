function U0 = set_ic(dof, u_val, v_val, p_val)
% SET_IC Generates the initial condition vector (vectorised).

    total_dof = dof.ndof;
    U0 = zeros(total_dof, 1);

    % Vectorised assignment using the DOF map directly
    U0(dof.node(:, 1)) = u_val;
    U0(dof.node(:, 2)) = v_val;
    U0(dof.node(:, 3)) = p_val;

    fprintf('Initial conditions set: u=%.3f, v=%.3f, p=%.3f\n', u_val, v_val, p_val);
end