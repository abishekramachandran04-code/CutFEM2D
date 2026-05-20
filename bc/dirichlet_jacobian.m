function [J_mod, R_mod] = dirichlet_jacobian(J, R, bc_dofs, bc_vals, U_k)
% DIRICHLET_JACOBIAN Applies essential BCs for Newton-Raphson (vectorised).
% Forces the increment dU to exactly match the required boundary step.

    bc_dofs = bc_dofs(:);
    bc_vals = bc_vals(:);

    J_mod = J;
    R_mod = R;

    % Zero out constrained rows in one shot
    J_mod(bc_dofs, :) = 0;

    % Set diagonal to 1 for all constrained DOFs
    J_mod = J_mod + sparse(bc_dofs, bc_dofs, ones(length(bc_dofs),1), size(J,1), size(J,2));

    % Set residual so that dU drives U_k to bc_vals
    R_mod(bc_dofs) = U_k(bc_dofs) - bc_vals;
end