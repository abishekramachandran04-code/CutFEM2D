function [K, F] = dirichlet(K, F, bc_dofs, bc_vals)
    % DIRICHLET Applies essential boundary conditions via fast sparse multiplication.
    
    bc_dofs = bc_dofs(:);
    bc_vals = bc_vals(:);
    N = size(K, 1);
    
    % 1. Modify the right-hand side force vector
    F = F - K(:, bc_dofs) * bc_vals;
    F(bc_dofs) = bc_vals;
    
    % 2. Build the Diagonal Filter Matrix (P)
    % This creates a vector of 1s, but puts a 0 at every boundary DOF.
    filter_vec = ones(N, 1);
    filter_vec(bc_dofs) = 0;
    
    % spdiags builds a sparse diagonal matrix instantly
    P = spdiags(filter_vec, 0, N, N);
    
    % 3. Zero out the rows and columns (The Magic Step)
    % P * K zeros the rows.
    % (P * K) * P zeros the columns.
    K = P * K * P;
    
    % 4. Put 1s back on the main diagonal for the constrained DOFs
    bc_vec = zeros(N, 1);
    bc_vec(bc_dofs) = 1;
    K = K + spdiags(bc_vec, 0, N, N);
end