function [L2_error, Linf_error] = calc_errors(mesh, conn, edof, dof, U, target_dof, exact_sol_func)
    % CALC_ERRORS Computes the continuous L2 and L_inf error norms over the domain.
    
    L2_error_sq = 0; 
    Linf_error = 0;
    
    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    ndpn = size(dof.node, 2);
    
    % Loop over each element type
    for t = 1:length(elem_types)
        type = elem_types{t};
        if isempty(conn.(type))
            continue;
        end
        
        elems = conn.(type);
        edofs = edof.(type);
        ne = size(elems, 1);
        nen = size(elems, 2);
        
        [xi, eta, weight] = gauss_quadrature(type);
        num_gp = length(weight);
        
        % Element loop
        for e = 1:ne
            node_ids = elems(e, :);
            node_coords = mesh.nodes(node_ids, 1:2);
            elem_dof = edofs(e, :);
            
            % Extract the nodal solutions for this specific element and target DOF
            u_local = zeros(nen, 1);
            for n = 1:nen
                dof_idx = (n-1)*ndpn + target_dof;
                u_local(n) = U(elem_dof(dof_idx));
            end
            
            % Gauss Quadrature Loop
            for q = 1:num_gp
                [N, dN_dxi] = shape_funcs(xi(q), eta(q), type);
                [~, detJ] = jacobian(dN_dxi, node_coords);
                
                % Interpolate the numerical solution at the Gauss point
                % N is [nen x 1], u_local is [nen x 1]. Dot product gives the scalar value.
                u_num_gp = N' * u_local;
                
                % Find the physical (x, y) coordinates of the Gauss point
                x_gp = N' * node_coords(:, 1);
                y_gp = N' * node_coords(:, 2);
                
                % Evaluate the exact analytical solution at (x_gp, y_gp)
                u_exact_gp = exact_sol_func(x_gp, y_gp);
                
                % Calculate point error
                err_val = abs(u_num_gp - u_exact_gp);
                
                % Accumulate the integrated squared L2 error
                L2_error_sq = L2_error_sq + (err_val^2) * detJ * weight(q);
                
                % Track the maximum L_infinity error
                if err_val > Linf_error
                    Linf_error = err_val;
                end
            end
        end
    end
    
    % Final L2 Norm
    L2_error = sqrt(L2_error_sq);
end