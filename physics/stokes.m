function [I, J, V, F] = stokes(mesh, conn, edof, dof, Re, f_source)
    % STOKES Assembles the 2D Stokes flow equations with PSPG stabilization.
    
    total_dof = dof.ndof;
    ndpn = 3; % u, v, p
    F = zeros(total_dof, 1);
    
    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    
    % 1. Preallocate triplet arrays
    num_entries = 0;
    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isempty(conn.(type))
            ne = size(conn.(type), 1);
            nen = size(conn.(type), 2);
            num_entries = num_entries + ne * (nen * ndpn)^2;
        end
    end
    
    I = zeros(num_entries, 1); J = zeros(num_entries, 1); V = zeros(num_entries, 1);
    idx = 1;
    
    % 2. Element Assembly Loop
    for t = 1:length(elem_types)
        type = elem_types{t};
        if isempty(conn.(type)), continue; end
        
        elems = conn.(type);
        edofs = edof.(type);
        ne = size(elems, 1);
        nen = size(elems, 2);
        
        [xi, eta, weight] = gauss_quadrature(type);
        num_gp = length(weight);
        
        for e = 1:ne
            node_ids = elems(e, :);
            node_coords = mesh.nodes(node_ids, 1:2);
            elem_dof = edofs(e, :);
            
            Ke = zeros(nen * ndpn, nen * ndpn);
            Fe = zeros(nen * ndpn, 1);
            
            % --- A. Calculate Element Area and Perimeter for PSPG Stabilization ---
            Ae = 0;
            for q = 1:num_gp
                [~, dN_dxi] = shape_funcs(xi(q), eta(q), type);
                [~, detJ] = jacobian(dN_dxi, node_coords);
                Ae = Ae + detJ * weight(q);
            end
            
            % Calculate Perimeter (P) dynamically from corner nodes
            if contains(type, 'tri')
                corners = node_coords(1:3, :);
            else
                corners = node_coords(1:4, :);
            end
            
            % Shift corners to compute edge vectors (1->2, 2->3, 3->4, 4->1)
            next_corners = [corners(2:end, :); corners(1, :)];
            edge_lengths = sqrt(sum((next_corners - corners).^2, 2));
            P = sum(edge_lengths);
            
            % Compute stabilization parameter tau using exact formula: h = 4A/P
            h_e = (4 * Ae) / P; 
            tau = (h_e^2 * Re) / 4;
            
            % --- B. Gauss Integration Loop ---
            is_quadratic = (nen == 6 || nen == 9);
            
            for q = 1:num_gp
                [N, dN_dxi] = shape_funcs(xi(q), eta(q), type);
                [dN_dx, detJ] = jacobian(dN_dxi, node_coords);
                dV = detJ * weight(q);
                
                % Explicitly build the local Jacobian matrix to get its Inverse components
                J_mat = dN_dxi * node_coords; % [cite: 50-52]
                J_star = inv(J_mat); % [cite: 56]
                J11_s = J_star(1,1); J12_s = J_star(1,2);
                J21_s = J_star(2,1); J22_s = J_star(2,2);
                
                % Fetch local second derivatives if using tri6 or quad9
                if is_quadratic
                    d2N_local = shape_funcs_2nd(xi(q), eta(q), type);
                end
                
                % Physical coordinates for body forces
                x_gp = N' * node_coords(:, 1);
                y_gp = N' * node_coords(:, 2);
                fx = f_source{1}(x_gp, y_gp); fy = f_source{2}(x_gp, y_gp);
                
                % Build the 3n x 3n element matrix block by block
                for a = 1:nen
                    for b = 1:nen
                        r_u = (a-1)*ndpn + 1; r_v = (a-1)*ndpn + 2; r_p = (a-1)*ndpn + 3;
                        c_u = (b-1)*ndpn + 1; c_v = (b-1)*ndpn + 2; c_p = (b-1)*ndpn + 3;
                        
                        Na = N(a); Nb = N(b);
                        dNa_dx = dN_dx(1, a); dNa_dy = dN_dx(2, a);
                        dNb_dx = dN_dx(1, b); dNb_dy = dN_dx(2, b);
                        
                        % Viscous Laplacian (Momentum)
                        visc = (1/Re) * (dNa_dx * dNb_dx + dNa_dy * dNb_dy);
                        
                        % Pressure Gradient (-B^T)
                        grad_x = -dNa_dx * Nb;
                        grad_y = -dNa_dy * Nb;
                        
                        % --- EXACT QUADRATIC PSPG FORMULATION ---
                        laplacian_Nb = 0;
                        if is_quadratic
                            d2Nb_dxi2 = d2N_local(1, b);
                            d2Nb_deta2 = d2N_local(2, b);
                            d2Nb_dxideta = d2N_local(3, b);
                            
                            % Map to physical domain using inverse Jacobian [cite: 112, 113]
                            d2Nb_dx2 = (J11_s^2)*d2Nb_dxi2 + 2*J11_s*J12_s*d2Nb_dxideta + (J12_s^2)*d2Nb_deta2;
                            d2Nb_dy2 = (J21_s^2)*d2Nb_dxi2 + 2*J21_s*J22_s*d2Nb_dxideta + (J22_s^2)*d2Nb_deta2;
                            laplacian_Nb = d2Nb_dx2 + d2Nb_dy2;
                        end
                        
                        % Velocity Divergence (B) with full stabilization [cite: 175, 176]
                        div_x = Na * dNb_dx - (tau / Re) * dNa_dx * laplacian_Nb;
                        div_y = Na * dNb_dy - (tau / Re) * dNa_dy * laplacian_Nb;
                        
                        % Pressure-Pressure Stabilization (C)
                        pspg = tau * (dNa_dx * dNb_dx + dNa_dy * dNb_dy);
                        
                        % Assemble into local Ke
                        Ke(r_u, c_u) = Ke(r_u, c_u) + visc * dV;
                        Ke(r_v, c_v) = Ke(r_v, c_v) + visc * dV;
                        
                        Ke(r_u, c_p) = Ke(r_u, c_p) + grad_x * dV;
                        Ke(r_v, c_p) = Ke(r_v, c_p) + grad_y * dV;
                        
                        Ke(r_p, c_u) = Ke(r_p, c_u) + div_x * dV;
                        Ke(r_p, c_v) = Ke(r_p, c_v) + div_y * dV;
                        
                        Ke(r_p, c_p) = Ke(r_p, c_p) + pspg * dV;
                    end
                    
                    % Assemble local Fe
                    r_u = (a-1)*ndpn + 1; 
                    r_v = (a-1)*ndpn + 2; 
                    r_p = (a-1)*ndpn + 3;
                    
                    % Standard Galerkin body forces for Momentum [cite: 17]
                    Fe(r_u) = Fe(r_u) + Na * fx * dV;
                    Fe(r_v) = Fe(r_v) + Na * fy * dV;
                    
                    % --- EXACT PSPG RIGHT-HAND SIDE ---
                    % Must include the stabilization body force term for Continuity 
                    Fe(r_p) = Fe(r_p) + tau * (dNa_dx * fx + dNa_dy * fy) * dV;
                end
            end
            
            % --- C. Scatter to Global Arrays ---
            F(elem_dof) = F(elem_dof) + Fe;
            for row = 1:(nen*ndpn)
                for col = 1:(nen*ndpn)
                    I(idx) = elem_dof(row);
                    J(idx) = elem_dof(col);
                    V(idx) = Ke(row, col);
                    idx = idx + 1;
                end
            end
        end
    end
end