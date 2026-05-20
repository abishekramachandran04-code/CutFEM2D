function [I, J, V, F] = navierstokes_cutfem(mesh, conn, edof, dof, Re, dt, U_k, U_n, f_source, phi, classification, ghost_info, params)
% NAVIERSTOKES_CUTFEM Assembly of Navier-Stokes with CutFEM geometry logic.

    total_dof = dof.ndof;
    ndpn = 3;
    F_cpu = zeros(total_dof, 1);
    I_all = []; J_all = []; V_all = [];

    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};

    % CPU data for CUT + Ghost Penalty (sequential)
    U_k_cpu = U_k(:);
    U_n_cpu = U_n(:);
    nodes_cpu = mesh.nodes(:,1:2);
    % GPU data for FULL_FLUID (vectorized)
    U_k_gpu = gpuArray(U_k_cpu);
    U_n_gpu = gpuArray(U_n_cpu);
    nodes_gpu = gpuArray(nodes_cpu);

    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isfield(conn, type) || isempty(conn.(type)), continue; end
        
        elems = conn.(type);
        edofs = edof.(type);
        nen = size(elems, 2);
        ndof_e = nen * ndpn;
        
        fluid_idx = classification.(type).FULL_FLUID;
        cut_idx = classification.(type).CUT;
        
        % ---------------------------------------------------------
        % 1. Vectorized Assembly for FULL_FLUID elements
        % ---------------------------------------------------------
        if ~isempty(fluid_idx)
            fluid_elems = elems(fluid_idx, :);
            fluid_edofs = edofs(fluid_idx, :);
            ne = length(fluid_idx);
            
            [xi, eta, wt] = gauss_quadrature(type);
            nGP = length(wt);
            is_quad = (nen == 6 || nen == 9);
            
            elems_g = gpuArray(fluid_elems);
            edofs_g = gpuArray(fluid_edofs);
            idx_f = reshape(elems_g', [], 1);
            coords = permute(reshape(nodes_gpu(idx_f, :), nen, ne, 2), [1 3 2]);
            ed_f = reshape(edofs_g', [], 1);
            Uk_l = reshape(U_k_gpu(ed_f), ndof_e, ne);
            Un_l = reshape(U_n_gpu(ed_f), ndof_e, ne);
            
            uk = Uk_l(1:3:end, :); vk = Uk_l(2:3:end, :);
            un = Un_l(1:3:end, :); vn = Un_l(2:3:end, :);
            
            theta = 0.5;
            u_adv = theta * uk + (1 - theta) * un;
            v_adv = theta * vk + (1 - theta) * vn;
            
            Ae = zeros(1, 1, ne, 'gpuArray');
            for q = 1:nGP
                [~, dNq] = shape_funcs(xi(q), eta(q), type);
                Jq = pagemtimes(gpuArray(dNq), coords);
                dJq = Jq(1,1,:).*Jq(2,2,:) - Jq(1,2,:).*Jq(2,1,:);
                Ae = Ae + dJq * wt(q);
            end
            
            if contains(type,'tri'), nc=3; else, nc=4; end
            corn = coords(1:nc, :, :);
            nxt = [corn(2:end,:,:); corn(1,:,:)];
            P_e = sum(sqrt(sum((nxt - corn).^2, 2)), 1);
            h_e = 4*Ae ./ P_e;
            
            if is_quad, p_order = 2; else, p_order = 1; end
            h_eff = h_e / p_order;
            
            um = reshape(mean(u_adv,1), 1,1,ne);
            vm = reshape(mean(v_adv,1), 1,1,ne);
            umag = sqrt(um.^2 + vm.^2);
            tau_pspg = 1 ./ sqrt((2/dt)^2 + (2*umag./h_eff).^2 + (4./(Re*h_eff.^2)).^2);
            
            % --- Dynamic SUPG (Transient-Aware Franca-Frey) for High-Re Flows ---
            % Re_h = (umag .* h_eff) ./ (2 * (1/Re));
            % xi_spatial = min(Re_h / 3.0, 1.0);
            % CFL = (umag .* dt) ./ h_eff;
            % xi_temporal = min(CFL, 1.0); % Scales SUPG down if time-step is very tight
            % tau_supg = tau_pspg .* (xi_spatial .* xi_temporal);
            
            % Manual decoupled SUPG for high-fidelity Low-Re Transient flows
            tau_supg = tau_pspg * 0.05;
            
            Ke = zeros(ndof_e, ndof_e, ne, 'gpuArray');
            Fe = zeros(ndof_e, 1, ne, 'gpuArray');
            iu = 1:3:ndof_e; iv = 2:3:ndof_e; ip = 3:3:ndof_e;
            
            for q = 1:nGP
                [N, dNxi] = shape_funcs(xi(q), eta(q), type);
                N = gpuArray(N(:)); dNxi = gpuArray(dNxi);
                Jm = pagemtimes(dNxi, coords);
                dJ = Jm(1,1,:).*Jm(2,2,:) - Jm(1,2,:).*Jm(2,1,:);
                
                iJ = zeros(2,2,ne,'gpuArray');
                iJ(1,1,:) =  Jm(2,2,:)./dJ; iJ(1,2,:) = -Jm(1,2,:)./dJ;
                iJ(2,1,:) = -Jm(2,1,:)./dJ; iJ(2,2,:) =  Jm(1,1,:)./dJ;
                
                dNx = pagemtimes(iJ, repmat(dNxi,1,1,ne));
                dV = dJ * wt(q);
                
                if is_quad
                    d2 = gpuArray(shape_funcs_2nd(xi(q), eta(q), type));
                    s11 = iJ(1,1,:); s12 = iJ(1,2,:); s21 = iJ(2,1,:); s22 = iJ(2,2,:);
                    lap = (s11.^2 + s21.^2).*d2(1,:) + 2*(s11.*s12 + s21.*s22).*d2(3,:) + (s12.^2 + s22.^2).*d2(2,:);
                else
                    lap = zeros(1, nen, ne, 'gpuArray');
                end
                
                xg = sum(N .* coords(:,1,:), 1); yg = sum(N .* coords(:,2,:), 1);
                fx = f_source{1}(xg, yg); fy = f_source{2}(xg, yg);
                
                ugk = reshape(N'*u_adv, 1,1,ne); vgk = reshape(N'*v_adv, 1,1,ne);
                ugn = reshape(N'*un, 1,1,ne); vgn = reshape(N'*vn, 1,1,ne);
                
                Ladv = ugk.*dNx(1,:,:) + vgk.*dNx(2,:,:);
                Lcol = permute(Ladv, [2 1 3]); Nt = reshape(N', 1, nen);
                dNxC = permute(dNx(1,:,:), [2 1 3]); dNyC = permute(dNx(2,:,:), [2 1 3]);
                
                M0 = N * Nt;
                Msupg = tau_supg .* pagemtimes(Lcol, Nt);
                Kvisc = (1/Re) * pagemtimes(permute(dNx,[2 1 3]), dNx);
                Kadv = pagemtimes(N, Ladv);
                Ksadv = tau_supg .* pagemtimes(Lcol, Ladv);
                Ksdif = tau_supg .* (-(1/Re)) .* pagemtimes(Lcol, lap);
                
                K_adv_diff = Kvisc + Kadv + Ksadv + Ksdif;
                Kuu = (M0 + Msupg + dt*theta*K_adv_diff) .* dV;
                Kup = (dt*(-1)) .* pagemtimes(dNxC, Nt) .* dV;
                Kvp = (dt*(-1)) .* pagemtimes(dNyC, Nt) .* dV;
                
                Mpu_x = tau_pspg .* pagemtimes(dNxC, Nt);
                Mpu_y = tau_pspg .* pagemtimes(dNyC, Nt);
                Dx = pagemtimes(N, dNx(1,:,:)); Dy = pagemtimes(N, dNx(2,:,:));
                Dpx = tau_pspg .* pagemtimes(dNxC, Ladv); Dpy = tau_pspg .* pagemtimes(dNyC, Ladv);
                Pvx = -(tau_pspg/Re) .* pagemtimes(dNxC, lap); Pvy = -(tau_pspg/Re) .* pagemtimes(dNyC, lap);
                
                Kpu = (Mpu_x + dt*Dx + dt*theta*(Dpx + Pvx)) .* dV;
                Kpv = (Mpu_y + dt*Dy + dt*theta*(Dpy + Pvy)) .* dV;
                Kpp = (dt .* tau_pspg) .* pagemtimes(permute(dNx,[2 1 3]), dNx) .* dV;
                
                Ke(iu,iu,:) = Ke(iu,iu,:) + Kuu; Ke(iv,iv,:) = Ke(iv,iv,:) + Kuu;
                Ke(iu,ip,:) = Ke(iu,ip,:) + Kup; Ke(iv,ip,:) = Ke(iv,ip,:) + Kvp;
                Ke(ip,iu,:) = Ke(ip,iu,:) + Kpu; Ke(ip,iv,:) = Ke(ip,iv,:) + Kpv;
                Ke(ip,ip,:) = Ke(ip,ip,:) + Kpp;
                
                un_reshaped = reshape(Un_l(1:3:end,:), nen, 1, ne);
                vn_reshaped = reshape(Un_l(2:3:end,:), nen, 1, ne);
                
                explicit_u = pagemtimes(K_adv_diff, un_reshaped) .* (-dt * (1-theta));
                explicit_v = pagemtimes(K_adv_diff, vn_reshaped) .* (-dt * (1-theta));
                
                explicit_p = pagemtimes(Dpx + Pvx, un_reshaped) + pagemtimes(Dpy + Pvy, vn_reshaped);
                explicit_p = explicit_p .* (-dt * (1-theta));
                
                NpL = N + tau_supg .* Lcol;
                Feu = (NpL .* (dt*fx + ugn) + explicit_u) .* dV;
                Fev = (NpL .* (dt*fy + vgn) + explicit_v) .* dV;
                Fep = (tau_pspg .* (dt*(dNxC.*fx + dNyC.*fy) + (dNxC.*ugn + dNyC.*vgn)) + explicit_p) .* dV;
                
                Fe(iu,1,:) = Fe(iu,1,:) + Feu; Fe(iv,1,:) = Fe(iv,1,:) + Fev; Fe(ip,1,:) = Fe(ip,1,:) + Fep;
            end
            
            r0 = repmat((1:ndof_e)', 1, ndof_e); c0 = repmat(1:ndof_e, ndof_e, 1);
            I_t = gather(edofs_g(:, r0(:))');
            J_t = gather(edofs_g(:, c0(:))');
            V_t = gather(reshape(Ke, ndof_e^2, ne));
            I_all = [I_all; I_t(:)];
            J_all = [J_all; J_t(:)];
            V_all = [V_all; V_t(:)];
            
            ed_cpu = gather(edofs_g');
            fe_cpu = gather(reshape(Fe, ndof_e, ne));
            F_cpu = F_cpu + accumarray(ed_cpu(:), fe_cpu(:), [total_dof, 1]);
        end
        
        % ---------------------------------------------------------
        % 2. Sequential Assembly for CUT elements
        % ---------------------------------------------------------
        if ~isempty(cut_idx)
            is_quad = (nen == 6 || nen == 9);
            iu = 1:3:ndof_e; iv = 2:3:ndof_e; ip = 3:3:ndof_e;
            
            theta = 0.5;
            u_adv = theta * U_k_cpu(dof.node(:,1)) + (1-theta) * U_n_cpu(dof.node(:,1));
            v_adv = theta * U_k_cpu(dof.node(:,2)) + (1-theta) * U_n_cpu(dof.node(:,2));
            
            for i = 1:length(cut_idx)
                e_global = cut_idx(i);
                nodes_e = elems(e_global, :);
                edofs_e = edofs(e_global, :);
                coords_e = nodes_cpu(nodes_e, :);
                phi_e = phi(nodes_e);
                
                Uk_e = U_k_cpu(edofs_e); Un_e = U_n_cpu(edofs_e);
                uk_e = Uk_e(iu); vk_e = Uk_e(iv);
                un_e = Un_e(iu); vn_e = Un_e(iv);
                
                Ke = zeros(ndof_e, ndof_e);
                Fe = zeros(ndof_e, 1);
                
                % Volume Integration
                subtriangles = subtriangulate_cut_element(coords_e, phi_e, type);
                [W_vol, xi_vol, eta_vol] = cut_quadrature(subtriangles, coords_e, type, 3);
                
                % Approximate h_e for SUPG and Nitsche using PARENT element area
                if contains(type,'tri'), nc=3; else, nc=4; end
                corn = coords_e(1:nc, :);
                P_e = sum(sqrt(sum((corn([2:end,1],:) - corn).^2, 2)));
                
                v1 = corn(2,:) - corn(1,:);
                v2 = corn(3,:) - corn(1,:);
                Ae_tot = 0.5 * abs(v1(1)*v2(2) - v1(2)*v2(1));
                if nc == 4
                    v3 = corn(3,:) - corn(4,:);
                    v4 = corn(1,:) - corn(4,:);
                    Ae_tot = Ae_tot + 0.5 * abs(v3(1)*v4(2) - v3(2)*v4(1));
                end
                
                h_e = 4 * Ae_tot / P_e;
                if h_e < 1e-12, h_e = 1e-12; end % safety
                
                if is_quad, p_order = 2; else, p_order = 1; end
                h_eff = h_e / p_order;
                
                um = mean(u_adv(nodes_e)); vm = mean(v_adv(nodes_e)); umag = sqrt(um^2 + vm^2);
                tau_pspg = 1 / sqrt((2/dt)^2 + (2*umag/h_eff)^2 + (4/(Re*h_eff^2))^2);
                
                % --- Dynamic SUPG (Transient-Aware Franca-Frey) for High-Re Flows ---
                % Re_h = (umag * h_eff) / (2 * (1/Re));
                % xi_spatial = min(Re_h / 3.0, 1.0);
                % CFL = (umag * dt) / h_eff;
                % xi_temporal = min(CFL, 1.0); % Scales SUPG down if time-step is very tight
                % tau_supg = tau_pspg * (xi_spatial * xi_temporal);
                
                % Manual decoupled SUPG for high-fidelity Low-Re Transient flows
                tau_supg = tau_pspg * 0.05;
                
                for q = 1:length(W_vol)
                    [N, dNxi] = shape_funcs(xi_vol(q), eta_vol(q), type);
                    Jm = dNxi * coords_e;
                    dJ = Jm(1,1)*Jm(2,2) - Jm(1,2)*Jm(2,1);
                    iJ = [Jm(2,2), -Jm(1,2); -Jm(2,1), Jm(1,1)] / dJ;
                    dNx = iJ * dNxi;
                    dV = W_vol(q);
                    
                    if is_quad
                        d2 = shape_funcs_2nd(xi_vol(q), eta_vol(q), type);
                        s11=iJ(1,1); s12=iJ(1,2); s21=iJ(2,1); s22=iJ(2,2);
                        lap = (s11^2 + s21^2)*d2(1,:) + 2*(s11*s12 + s21*s22)*d2(3,:) + (s12^2 + s22^2)*d2(2,:);
                    else
                        lap = zeros(1, nen);
                    end
                    
                    xg = dot(N, coords_e(:,1)); yg = dot(N, coords_e(:,2));
                    fx = f_source{1}(xg, yg); fy = f_source{2}(xg, yg);
                    
                    ugk = dot(N, u_adv(nodes_e)); vgk = dot(N, v_adv(nodes_e));
                    ugn = dot(N, un_e); vgn = dot(N, vn_e);
                    
                    Ladv = ugk*dNx(1,:) + vgk*dNx(2,:);
                    Lcol = Ladv'; Nt = N';
                    dNxC = dNx(1,:)'; dNyC = dNx(2,:)';
                    
                    M0 = N * Nt;
                    Msupg = tau_supg * (Lcol * Nt);
                    Kvisc = (1/Re) * (dNx' * dNx);
                    Kadv = N * Ladv;
                    Ksadv = tau_supg * (Lcol * Ladv);
                    Ksdif = tau_supg * (-1/Re) * (Lcol * lap);
                    
                    K_adv_diff = Kvisc + Kadv + Ksadv + Ksdif;
                    Kuu = (M0 + Msupg + dt*theta*K_adv_diff) * dV;
                    Kup = -dt * (dNxC * Nt) * dV;
                    Kvp = -dt * (dNyC * Nt) * dV;
                    
                    Mpu_x = tau_pspg * (dNxC * Nt); Mpu_y = tau_pspg * (dNyC * Nt);
                    Dx = N * dNx(1,:); Dy = N * dNx(2,:);
                    Dpx = tau_pspg * (dNxC * Ladv); Dpy = tau_pspg * (dNyC * Ladv);
                    Pvx = -(tau_pspg/Re) * (dNxC * lap); Pvy = -(tau_pspg/Re) * (dNyC * lap);
                    
                    Kpu = (Mpu_x + dt*Dx + dt*theta*(Dpx + Pvx)) * dV;
                    Kpv = (Mpu_y + dt*Dy + dt*theta*(Dpy + Pvy)) * dV;
                    Kpp = (dt * tau_pspg) * (dNx' * dNx) * dV;
                    
                    Ke(iu,iu) = Ke(iu,iu) + Kuu; Ke(iv,iv) = Ke(iv,iv) + Kuu;
                    Ke(iu,ip) = Ke(iu,ip) + Kup; Ke(iv,ip) = Ke(iv,ip) + Kvp;
                    Ke(ip,iu) = Ke(ip,iu) + Kpu; Ke(ip,iv) = Ke(ip,iv) + Kpv;
                    Ke(ip,ip) = Ke(ip,ip) + Kpp;
                    
                    explicit_u = K_adv_diff * un_e .* (-dt * (1-theta));
                    explicit_v = K_adv_diff * vn_e .* (-dt * (1-theta));
                    
                    explicit_p = (Dpx + Pvx)*un_e + (Dpy + Pvy)*vn_e;
                    explicit_p = explicit_p .* (-dt * (1-theta));
                    
                    NpL = N + tau_supg * Lcol;
                    Fe(iu) = Fe(iu) + (NpL * (dt*fx + ugn) + explicit_u) * dV;
                    Fe(iv) = Fe(iv) + (NpL * (dt*fy + vgn) + explicit_v) * dV;
                    Fe(ip) = Fe(ip) + (tau_pspg * (dt*(dNxC*fx + dNyC*fy) + (dNxC*ugn + dNyC*vgn)) + explicit_p) * dV;
                end
                
                % Nitsche Boundary Assembly
                [W_line, xi_line, eta_line, nx, ny] = nitsche_line_quadrature(coords_e, phi_e, type, 3);
                for q = 1:length(W_line)
                    [N, dNxi] = shape_funcs(xi_line(q), eta_line(q), type);
                    Jm = dNxi * coords_e;
                    dJ = Jm(1,1)*Jm(2,2) - Jm(1,2)*Jm(2,1);
                    iJ = [Jm(2,2), -Jm(1,2); -Jm(2,1), Jm(1,1)] / dJ;
                    dNx = iJ * dNxi;
                    dL = W_line(q);
                    
                    n_x = nx(q); n_y = ny(q);
                    Nt = N'; dNxC = dNx(1,:)'; dNyC = dNx(2,:)';
                    
                    % Dynamically fetch cylinder boundary velocity (e.g., for oscillating cylinders)
                    if isfield(params, 'uD'), uD = params.uD; else, uD = 0; end
                    if isfield(params, 'vD'), vD = params.vD; else, vD = 0; end
                    
                    % Nitsche Consistency (Symmetric)
                    Kuu_nc = -dt * (1/Re) * (N * (dNx(1,:)*n_x + dNx(2,:)*n_y)) * dL;
                    Kuu_ns = -dt * (1/Re) * ((dNxC*n_x + dNyC*n_y) * Nt) * dL;
                    
                    % Penalty
                    Kuu_np = dt * (params.gamma_u / (Re * h_e)) * (N * Nt) * dL;
                    
                    Kuu = Kuu_nc + Kuu_ns + Kuu_np;
                    
                    Ke(iu,iu) = Ke(iu,iu) + Kuu; Ke(iv,iv) = Ke(iv,iv) + Kuu;
                    
                    % Nitsche Pressure-Velocity Continuity Symmetry
                    Kpu_nc = dt * (N * Nt * n_x) * dL;
                    Kpv_nc = dt * (N * Nt * n_y) * dL;
                    Kup_nc = dt * (N * Nt * n_x) * dL;
                    Kvp_nc = dt * (N * Nt * n_y) * dL;
                    
                    Ke(ip,iu) = Ke(ip,iu) + Kpu_nc; Ke(ip,iv) = Ke(ip,iv) + Kpv_nc;
                    Ke(iu,ip) = Ke(iu,ip) + Kup_nc; Ke(iv,ip) = Ke(iv,ip) + Kvp_nc;
                    
                    % RHS Boundary conditions (uD, vD)
                    Fe(iu) = Fe(iu) - dt * (1/Re) * (dNxC*n_x + dNyC*n_y) * uD * dL + dt * (params.gamma_u / (Re * h_e)) * N * uD * dL;
                    Fe(iv) = Fe(iv) - dt * (1/Re) * (dNxC*n_x + dNyC*n_y) * vD * dL + dt * (params.gamma_u / (Re * h_e)) * N * vD * dL;
                    Fe(ip) = Fe(ip) + dt * N * (uD*n_x + vD*n_y) * dL;
                end
                
                r0 = repmat((1:ndof_e)', 1, ndof_e); c0 = repmat(1:ndof_e, ndof_e, 1);
                I_all = [I_all; edofs_e(r0(:))'];
                J_all = [J_all; edofs_e(c0(:))'];
                V_all = [V_all; Ke(:)];
                F_cpu(edofs_e) = F_cpu(edofs_e) + Fe;
            end
        end
    end
    
    % ---------------------------------------------------------
    % 3. Ghost Penalty Assembly (Gradient Jump Formulation)
    % ---------------------------------------------------------
    if ~isempty(ghost_info) && (params.alpha_v > 0 || params.alpha_p > 0 || params.alpha_adv > 0)
        edges = ghost_info.edges;
        elem_L = ghost_info.elem_L;
        elem_R = ghost_info.elem_R;
        type_L = ghost_info.type_L;
        type_R = ghost_info.type_R;
        
        [xi_1d, w_1d] = get_1d_gauss_points_gp(3);
        
        for g = 1:size(edges, 1)
            eL = elem_L(g); eR = elem_R(g);
            tL = type_L{g}; tR = type_R{g};
            
            nodes_L = conn.(tL)(eL, :); nodes_R = conn.(tR)(eR, :);
            coords_L = nodes_cpu(nodes_L, :); coords_R = nodes_cpu(nodes_R, :);
            edofs_L = edof.(tL)(eL, :); edofs_R = edof.(tR)(eR, :);
            
            p1 = nodes_cpu(edges(g, 1), :); p2 = nodes_cpu(edges(g, 2), :);
            dx = p2(1) - p1(1); dy = p2(2) - p1(2);
            he = sqrt(dx^2 + dy^2);
            if he < 1e-12, continue; end
            nx = dy / he; ny = -dx / he; 
            
            um = 0.5 * (U_k_cpu(dof.node(edges(g,1),1)) + U_k_cpu(dof.node(edges(g,2),1)));
            vm = 0.5 * (U_k_cpu(dof.node(edges(g,1),2)) + U_k_cpu(dof.node(edges(g,2),2)));
            un_edge = abs(um * nx + vm * ny);
            umag = sqrt(um^2 + vm^2);
            tau_e = 1 / sqrt((2/dt)^2 + (2*umag/he)^2 + (4/(Re*he^2))^2);
            
            % Ghost penalty scaling must match ALL terms in the element matrix:
            %   Mass:    M/dt  ~ h^2/dt   → need gamma/h ~ h^2  → gamma ~ h^3
            %   Viscous: K_visc ~ 1/Re     → need gamma/h ~ dt/Re → gamma ~ dt*h/Re
            %   Advect:  K_adv  ~ |u|      → need gamma/h ~ dt*|u| → gamma ~ dt*h*|u|
            gamma_v = params.alpha_v * he^3 ...                     % mass-level (dominant for small dt)
                    + dt * params.alpha_v * he / Re ...              % viscous-level
                    + dt * params.alpha_adv * un_edge * he^2;        % advective-level
            gamma_p = params.alpha_p * he^3 ...                     % mass-level for pressure
                    + dt * params.alpha_p * tau_e * he;              % PSPG-level
            
            ndof_L = length(nodes_L)*3; ndof_R = length(nodes_R)*3;
            iu_L = 1:3:ndof_L; iv_L = 2:3:ndof_L; ip_L = 3:3:ndof_L;
            iu_R = 1:3:ndof_R; iv_R = 2:3:ndof_R; ip_R = 3:3:ndof_R;
            
            K_LL = zeros(ndof_L, ndof_L); K_RR = zeros(ndof_R, ndof_R);
            K_LR = zeros(ndof_L, ndof_R); K_RL = zeros(ndof_R, ndof_L);
            
            is_high_order = contains(tL, '9') || contains(tL, '6');
            
            for q = 1:length(w_1d)
                t = 0.5 * (1 + xi_1d(q));
                x_g = p1(1) + t * dx; y_g = p1(2) + t * dy;
                dL = (he / 2) * w_1d(q);
                
                [xi_L, eta_L] = inverse_mapping_gp(x_g, y_g, coords_L, tL);
                [~, dNxi_L] = shape_funcs(xi_L, eta_L, tL);
                Jm_L = dNxi_L * coords_L; dJ_L = Jm_L(1,1)*Jm_L(2,2) - Jm_L(1,2)*Jm_L(2,1);
                iJ_L = [Jm_L(2,2), -Jm_L(1,2); -Jm_L(2,1), Jm_L(1,1)] / dJ_L;
                dNx_L = iJ_L * dNxi_L;
                gLn = dNx_L(1,:) * nx + dNx_L(2,:) * ny;
                
                [xi_R, eta_R] = inverse_mapping_gp(x_g, y_g, coords_R, tR);
                [~, dNxi_R] = shape_funcs(xi_R, eta_R, tR);
                Jm_R = dNxi_R * coords_R; dJ_R = Jm_R(1,1)*Jm_R(2,2) - Jm_R(1,2)*Jm_R(2,1);
                iJ_R = [Jm_R(2,2), -Jm_R(1,2); -Jm_R(2,1), Jm_R(1,1)] / dJ_R;
                dNx_R = iJ_R * dNxi_R;
                gRn = dNx_R(1,:) * nx + dNx_R(2,:) * ny;
                
                if is_high_order
                    d2L = shape_funcs_2nd(xi_L, eta_L, tL);
                    vL = iJ_L * [nx; ny];
                    gLn2 = vL(1)^2 * d2L(1,:) + vL(2)^2 * d2L(2,:) + 2 * vL(1)*vL(2) * d2L(3,:);
                    
                    d2R = shape_funcs_2nd(xi_R, eta_R, tR);
                    vR = iJ_R * [nx; ny];
                    gRn2 = vR(1)^2 * d2R(1,:) + vR(2)^2 * d2R(2,:) + 2 * vR(1)*vR(2) * d2R(3,:);
                    
                    B_LL = (gLn' * gLn + he^2 * (gLn2' * gLn2)) * dL;
                    B_RR = (gRn' * gRn + he^2 * (gRn2' * gRn2)) * dL;
                    B_LR = -(gLn' * gRn + he^2 * (gLn2' * gRn2)) * dL;
                    B_RL = -(gRn' * gLn + he^2 * (gRn2' * gLn2)) * dL;
                else
                    B_LL = gLn' * gLn * dL; 
                    B_RR = gRn' * gRn * dL;
                    B_LR = -(gLn' * gRn) * dL; 
                    B_RL = -(gRn' * gLn) * dL;
                end
                
                K_LL(iu_L, iu_L) = K_LL(iu_L, iu_L) + gamma_v * B_LL; K_LL(iv_L, iv_L) = K_LL(iv_L, iv_L) + gamma_v * B_LL; K_LL(ip_L, ip_L) = K_LL(ip_L, ip_L) + gamma_p * B_LL;
                K_RR(iu_R, iu_R) = K_RR(iu_R, iu_R) + gamma_v * B_RR; K_RR(iv_R, iv_R) = K_RR(iv_R, iv_R) + gamma_v * B_RR; K_RR(ip_R, ip_R) = K_RR(ip_R, ip_R) + gamma_p * B_RR;
                K_LR(iu_L, iu_R) = K_LR(iu_L, iu_R) + gamma_v * B_LR; K_LR(iv_L, iv_R) = K_LR(iv_L, iv_R) + gamma_v * B_LR; K_LR(ip_L, ip_R) = K_LR(ip_L, ip_R) + gamma_p * B_LR;
                K_RL(iu_R, iu_L) = K_RL(iu_R, iu_L) + gamma_v * B_RL; K_RL(iv_R, iv_L) = K_RL(iv_R, iv_L) + gamma_v * B_RL; K_RL(ip_R, ip_L) = K_RL(ip_R, ip_L) + gamma_p * B_RL;
            end
            
            rLL = repmat((1:ndof_L)', 1, ndof_L); cLL = repmat(1:ndof_L, ndof_L, 1);
            I_all = [I_all; edofs_L(rLL(:))']; J_all = [J_all; edofs_L(cLL(:))']; V_all = [V_all; K_LL(:)];
            
            rRR = repmat((1:ndof_R)', 1, ndof_R); cRR = repmat(1:ndof_R, ndof_R, 1);
            I_all = [I_all; edofs_R(rRR(:))']; J_all = [J_all; edofs_R(cRR(:))']; V_all = [V_all; K_RR(:)];
            
            rLR = repmat((1:ndof_L)', 1, ndof_R); cLR = repmat(1:ndof_R, ndof_L, 1);
            I_all = [I_all; edofs_L(rLR(:))']; J_all = [J_all; edofs_R(cLR(:))']; V_all = [V_all; K_LR(:)];
            
            rRL = repmat((1:ndof_R)', 1, ndof_L); cRL = repmat(1:ndof_L, ndof_R, 1);
            I_all = [I_all; edofs_R(rRL(:))']; J_all = [J_all; edofs_L(cRL(:))']; V_all = [V_all; K_RL(:)];
        end
    end

    I = I_all;
    J = J_all;
    V = V_all;
    F = F_cpu;
end

function [xi_1d, w_1d] = get_1d_gauss_points_gp(n)
    if n == 1
        xi_1d = 0; w_1d = 2;
    elseif n == 2
        xi_1d = [-1/sqrt(3); 1/sqrt(3)]; w_1d = [1; 1];
    elseif n == 3
        xi_1d = [-sqrt(3/5); 0; sqrt(3/5)]; w_1d = [5/9; 8/9; 5/9];
    else
        error('Unsupported 1D Gauss points.');
    end
end

function [xi, eta] = inverse_mapping_gp(x, y, elem_nodes, elem_type)
    if strcmp(elem_type, 'quad4') || strcmp(elem_type, 'quad9')
        xi = 0.0; eta = 0.0;
    else
        xi = 1/3; eta = 1/3;
    end
    max_iter = 20; tol = 1e-10;
    for iter = 1:max_iter
        [N, dN_dxi] = shape_funcs(xi, eta, elem_type);
        x_curr = dot(N, elem_nodes(:, 1)); y_curr = dot(N, elem_nodes(:, 2));
        R = [x - x_curr; y - y_curr];
        if norm(R) < tol, break; end
        dx_dxi = dot(dN_dxi(1,:), elem_nodes(:, 1)); dy_dxi = dot(dN_dxi(1,:), elem_nodes(:, 2));
        dx_deta = dot(dN_dxi(2,:), elem_nodes(:, 1)); dy_deta = dot(dN_dxi(2,:), elem_nodes(:, 2));
        J = [dx_dxi, dx_deta; dy_dxi, dy_deta];
        delta = J \ R;
        xi = xi + delta(1); eta = eta + delta(2);
    end
end
