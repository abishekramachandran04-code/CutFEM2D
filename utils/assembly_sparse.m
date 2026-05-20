function K = assembly_sparse(I, J, V, total_dof)
    % ASSEMBLY_SPARSE Creates the global sparse stiffness matrix from triplets.
    % MATLAB automatically sums values at duplicate (I,J) indices, 
    % cleanly handling the overlapping contributions at shared nodes.
    
    K = sparse(I, J, V, total_dof, total_dof);
end