function [I, J, V, F] = poisson(mesh, conn, edof, dof, k_diff, f_source)
    % k_diff:   [1 x ndpn] vector of diffusion coefficients (e.g., [1.0, 0] for x-only)
    % f_source: [1 x ndpn] vector of source terms
    
    total_dof = dof.ndof;
    ndpn = size(dof.node, 2);
    F = zeros(total_dof, 1);
    
    % 1. Preallocate triplet arrays based on total element entries
    num_entries = 0;
    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isempty(conn.(type))
            ne = size(conn.(type), 1);
            nen = size(conn.(type), 2);
            % Ke size is now (nen*ndpn) x (nen*ndpn)
            num_entries = num_entries + ne * (nen * ndpn)^2; 
        end
    end
    
    I = zeros(num_entries, 1);
    J = zeros(num_entries, 1);
    V = zeros(num_entries, 1);
    
    idx = 1; % Triplet index counter
    
    % 2. Loop over each element type
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
        
% 3. Element integration loop
        for e = 1:ne
            node_ids = elems(e, :);
            node_coords = mesh.nodes(node_ids, 1:2); 
            elem_dof = edofs(e, :);
            
            Ke_scalar = zeros(nen, nen);
            Fe = zeros(nen * ndpn, 1); % Multi-DOF force vector
            
            % Gauss Quadrature Loop
            for q = 1:num_gp
                [N, dN_dxi] = shape_funcs(xi(q), eta(q), type);
                [dN_dx, detJ] = jacobian(dN_dxi, node_coords);
                
                % 1. Accumulate the scalar Stiffness Matrix
                B = dN_dx; 
                Ke_scalar = Ke_scalar + (B' * B) * detJ * weight(q);
                
                % 2. Find physical (x, y) coordinates of this Gauss point
                x_gp = N' * node_coords(:, 1);
                y_gp = N' * node_coords(:, 2);
                
                % 3. Evaluate and accumulate the Force Vector for each DOF
                for d = 1:ndpn
                    % Check if the user passed a cell array of functions, or a constant array
                    if iscell(f_source) && isa(f_source{d}, 'function_handle')
                        f_val = f_source{d}(x_gp, y_gp);
                    elseif isnumeric(f_source)
                        f_val = f_source(d);
                    else
                        f_val = 0;
                    end
                    
                    % Scatter the evaluated force into the correct DOF rows
                    for row = 1:nen
                        r_idx = (row-1)*ndpn + d;
                        Fe(r_idx) = Fe(r_idx) + f_val * N(row) * detJ * weight(q);
                    end
                end
            end
            
            % 4. Expand scalar Stiffness into the multi-DOF local matrix
            Ke = zeros(nen * ndpn, nen * ndpn);
            for d = 1:ndpn
                if k_diff(d) ~= 0
                    for row = 1:nen
                        for col = 1:nen
                            r_idx = (row-1)*ndpn + d;
                            c_idx = (col-1)*ndpn + d;
                            Ke(r_idx, c_idx) = k_diff(d) * Ke_scalar(row, col);
                        end
                    end
                end
            end
            
            % Scatter into global F directly
            F(elem_dof) = F(elem_dof) + Fe;
            
            % Scatter local Ke into triplet arrays
            for row = 1:(nen * ndpn)
                for col = 1:(nen * ndpn)
                    I(idx) = elem_dof(row);
                    J(idx) = elem_dof(col);
                    V(idx) = Ke(row, col);
                    idx = idx + 1;
                end
            end
        end
    end
end