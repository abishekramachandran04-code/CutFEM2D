function [K, F] = assembly(mesh, conn, edof, dof, physics_type, params)
    % ASSEMBLY Orchestrates the matrix building by calling the correct physics module.
    
    switch lower(physics_type)
        case 'poisson'
            % params = struct with k_diff array and f_source array
            k_diff = params.k;
            f_source = params.f;
            [I, J, V, F] = poisson(mesh, conn, edof, dof, k_diff, f_source);

        case 'stokes'
            Re = params.Re;
            f_source = params.f;
            [I, J, V, F] = stokes(mesh, conn, edof, dof, Re, f_source);
            
        case 'navier_stokes'
            % Placeholder for your ultimate target
            error('Navier-Stokes physics module not yet implemented.');
            
        otherwise
            error('Unsupported physics type: %s', physics_type);
    end
    
    % Generate the final sparse global stiffness matrix
    K = assembly_sparse(I, J, V, dof.ndof);
end