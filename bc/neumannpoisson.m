function F = neumannpoisson(mesh, edge_conn, dof, F, q_flux, target_dof)
    % NEUMANN Adds boundary flux to the global force vector using 1D integration.
    
    ne = size(edge_conn, 1);
    if ne == 0
        return;
    end
    
    % Dynamically determine if this is a linear (2) or quadratic (3) edge
    nen = size(edge_conn, 2); 
    
    % Select 1D Quadrature rule based on element order
    if nen == 2
        % 2-node linear edge (2-point rule)
        xi = [-1/sqrt(3); 1/sqrt(3)];
        weight = [1; 1];
    elseif nen == 3
        % 3-node quadratic edge (3-point rule)
        xi = [-sqrt(3/5); 0; sqrt(3/5)];
        weight = [5/9; 8/9; 5/9];
    else
        error('Unsupported number of edge nodes: %d', nen);
    end
    
    for e = 1:ne
        node_ids = edge_conn(e, :);
        coords = mesh.nodes(node_ids, 1:2);
        
        Fe = zeros(nen, 1);
        
        for q = 1:length(weight)
            % 1D Shape functions and local derivatives
            if nen == 2
                N = [0.5 * (1 - xi(q)); 
                     0.5 * (1 + xi(q))];
                dN_dxi = [-0.5; 0.5];
                
            elseif nen == 3
                % Gmsh standard ordering for 3-node lines: Node 1(-1), Node 2(1), Node 3(0)
                N = [0.5 * xi(q) * (xi(q) - 1);
                     0.5 * xi(q) * (xi(q) + 1);
                     1 - xi(q)^2];
                dN_dxi = [xi(q) - 0.5;
                          xi(q) + 0.5;
                          -2 * xi(q)];
            end
            
            % 1D Jacobian (ds/dxi) to transform length from parent to physical space
            dx_dxi = dN_dxi' * coords(:, 1);
            dy_dxi = dN_dxi' * coords(:, 2);
            detJ = sqrt(dx_dxi^2 + dy_dxi^2); 
            
            % Global coordinates of this specific integration point
            x_gp = N' * coords(:, 1);
            y_gp = N' * coords(:, 2);
            
            % Evaluate the flux function at the integration point
            if isa(q_flux, 'function_handle')
                q_val = q_flux(x_gp, y_gp);
            else
                q_val = q_flux;
            end
            
            % Accumulate local boundary force
            Fe = Fe + N * q_val * detJ * weight(q);
        end
        
        % Scatter to the global F vector
        for i = 1:nen
            n = node_ids(i);
            global_dof = dof.node(n, target_dof);
            F(global_dof) = F(global_dof) + Fe(i);
        end
    end
end