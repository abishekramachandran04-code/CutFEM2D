function F = neumannstokes(mesh, edge_conn, dof, F, P_nd, tau_nd)
    % NEUMANN Adds fluid pressure and shear traction to the global force vector.
    % Automatically handles curved boundaries by computing local normals.
    
    ne = size(edge_conn, 1);
    if ne == 0
        return;
    end
    
    % Dynamically determine if this is a linear (2) or quadratic (3) edge
    nen = size(edge_conn, 2); 
    
    % Select 1D Quadrature rule based on element order
    if nen == 2
        xi = [-1/sqrt(3); 1/sqrt(3)];
        weight = [1; 1];
    elseif nen == 3
        xi = [-sqrt(3/5); 0; sqrt(3/5)];
        weight = [5/9; 8/9; 5/9];
    else
        error('Unsupported number of edge nodes: %d', nen);
    end
    
    for e = 1:ne
        node_ids = edge_conn(e, :);
        coords = mesh.nodes(node_ids, 1:2);
        
        % Preallocate local force vectors for both u and v
        Fe_u = zeros(nen, 1);
        Fe_v = zeros(nen, 1);
        
        for q = 1:length(weight)
            % 1. 1D Shape functions and local derivatives
            if nen == 2
                N = [0.5 * (1 - xi(q)); 
                     0.5 * (1 + xi(q))];
                dN_dxi = [-0.5; 0.5];
                
            elseif nen == 3
                N = [0.5 * xi(q) * (xi(q) - 1);
                     0.5 * xi(q) * (xi(q) + 1);
                     1 - xi(q)^2];
                dN_dxi = [xi(q) - 0.5;
                          xi(q) + 0.5;
                          -2 * xi(q)];
            end
            
            % 2. 1D Jacobian (dx/dxi, dy/dxi) mapping
            dx_dxi = dN_dxi' * coords(:, 1);
            dy_dxi = dN_dxi' * coords(:, 2);
            detJ = sqrt(dx_dxi^2 + dy_dxi^2); 
            
            % 3. Calculate Unit Vectors dynamically for this Gauss point
            % Tangent vector (pointing strictly along the edge)
            t_x = dx_dxi / detJ;
            t_y = dy_dxi / detJ;
            
            % Outward Normal vector (rotated 90 degrees clockwise from tangent)
            n_x = dy_dxi / detJ;
            n_y = -dx_dxi / detJ;

            % ... [existing shape function and Jacobian code] ...
            
            % NEW: Calculate global physical coordinates of the Gauss point
            x_gp = N' * coords(:, 1);
            y_gp = N' * coords(:, 2);
            
            % NEW: Evaluate P and tau (handles both constants and functions)
            if isa(P_nd, 'function_handle')
                P_val = P_nd(x_gp, y_gp);
            else
                P_val = P_nd;
            end
            
            if isa(tau_nd, 'function_handle')
                tau_val = tau_nd(x_gp, y_gp);
            else
                tau_val = tau_nd;
            end
            
            % 4. Build the Traction Vector 
            % Pressure pushes strictly AGAINST the normal
            % Shear drags strictly ALONG the tangent
            % Build the Traction Vector dynamically using the evaluated values
            h_x = -P_val * n_x + tau_val * t_x;
            h_y = -P_val * n_y + tau_val * t_y;
            
            % 5. Accumulate local boundary force
            Fe_u = Fe_u + N * h_x * detJ * weight(q);
            Fe_v = Fe_v + N * h_y * detJ * weight(q);
        end
        
        % Scatter to the global F vector for both DOFs simultaneously
        for i = 1:nen
            n = node_ids(i);
            global_dof_u = dof.node(n, 1);
            global_dof_v = dof.node(n, 2);
            
            F(global_dof_u) = F(global_dof_u) + Fe_u(i);
            F(global_dof_v) = F(global_dof_v) + Fe_v(i);
        end
    end
end