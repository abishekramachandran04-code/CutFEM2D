function [I, J, V, F] = navierstokes(mesh, conn, edof, dof, Re, dt, U_k, U_n, f_source)
% NAVIERSTOKES  GPU-vectorised assembly of the transient 2D Navier-Stokes
% equations with SUPG/PSPG stabilisation (Picard linearisation).
%
% All element-level work is batched across every element of each type
% using pagemtimes, eliminating the scalar element loop entirely.
% Data lives on the GPU; the returned triplets (I,J,V) and F are gathered
% back to CPU so the caller can build a sparse matrix normally.

    total_dof = dof.ndof;
    ndpn = 3;
    F_cpu = zeros(total_dof, 1);

    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    I_all = []; J_all = []; V_all = [];

    U_k_g = gpuArray(U_k(:));
    U_n_g = gpuArray(U_n(:));
    nodes_g = gpuArray(mesh.nodes(:,1:2));

    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isfield(conn, type) || isempty(conn.(type)), continue; end

        elems  = conn.(type);
        edofs  = edof.(type);
        ne     = size(elems, 1);
        nen    = size(elems, 2);
        ndof_e = nen * ndpn;

        [xi, eta, wt] = gauss_quadrature(type);
        nGP = length(wt);
        is_quad = (nen == 6 || nen == 9);

        elems_g = gpuArray(elems);
        edofs_g = gpuArray(edofs);

        idx_f = reshape(elems_g', [], 1);
        crd_f = nodes_g(idx_f, :);
        coords = permute(reshape(crd_f, nen, ne, 2), [1 3 2]);

        ed_f  = reshape(edofs_g', [], 1);
        Uk_l  = reshape(U_k_g(ed_f), ndof_e, ne);
        Un_l  = reshape(U_n_g(ed_f), ndof_e, ne);

        theta = 0.5; % Crank-Nicolson parameter
        
        uk = Uk_l(1:3:end, :);   vk = Uk_l(2:3:end, :);
        un = Un_l(1:3:end, :);   vn = Un_l(2:3:end, :);
        
        % Time-centered advecting velocity to eliminate non-linear temporal viscosity
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
        nxt  = [corn(2:end,:,:); corn(1,:,:)];
        P_e  = sum(sqrt(sum((nxt - corn).^2, 2)), 1);
        h_e  = 4*Ae ./ P_e;

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
        Fe = zeros(ndof_e, 1,      ne, 'gpuArray');
        iu = 1:3:ndof_e;   iv = 2:3:ndof_e;   ip = 3:3:ndof_e;

        for q = 1:nGP
            [N, dNxi] = shape_funcs(xi(q), eta(q), type);
            N    = gpuArray(N(:));
            dNxi = gpuArray(dNxi);

            Jm = pagemtimes(dNxi, coords);
            dJ = Jm(1,1,:).*Jm(2,2,:) - Jm(1,2,:).*Jm(2,1,:);

            iJ = zeros(2,2,ne,'gpuArray');
            iJ(1,1,:) =  Jm(2,2,:)./dJ;
            iJ(1,2,:) = -Jm(1,2,:)./dJ;
            iJ(2,1,:) = -Jm(2,1,:)./dJ;
            iJ(2,2,:) =  Jm(1,1,:)./dJ;

            dNx = pagemtimes(iJ, repmat(dNxi,1,1,ne));
            dV  = dJ * wt(q);

            if is_quad
                d2  = gpuArray(shape_funcs_2nd(xi(q), eta(q), type));
                s11 = iJ(1,1,:); s12 = iJ(1,2,:);
                s21 = iJ(2,1,:); s22 = iJ(2,2,:);
                lap = (s11.^2 + s21.^2).*d2(1,:) ...
                    + 2*(s11.*s12 + s21.*s22).*d2(3,:) ...
                    + (s12.^2 + s22.^2).*d2(2,:);
            else
                lap = zeros(1, nen, ne, 'gpuArray');
            end

            xg = sum(N .* coords(:,1,:), 1);
            yg = sum(N .* coords(:,2,:), 1);
            fx = f_source{1}(xg, yg);
            fy = f_source{2}(xg, yg);

            ugk = reshape(N'*u_adv, 1,1,ne);
            vgk = reshape(N'*v_adv, 1,1,ne);
            ugn = reshape(N'*un, 1,1,ne);
            vgn = reshape(N'*vn, 1,1,ne);

            Ladv = ugk.*dNx(1,:,:) + vgk.*dNx(2,:,:);

            Lcol  = permute(Ladv, [2 1 3]);
            Nt    = reshape(N', 1, nen);
            dNxC  = permute(dNx(1,:,:), [2 1 3]);
            dNyC  = permute(dNx(2,:,:), [2 1 3]);

            M0    = N * Nt;
            Msupg = tau_supg .* pagemtimes(Lcol, Nt);
            Kvisc = (1/Re) * pagemtimes(permute(dNx,[2 1 3]), dNx);
            Kadv  = pagemtimes(N, Ladv);
            Ksadv = tau_supg .* pagemtimes(Lcol, Ladv);
            Ksdif = tau_supg .* (-(1/Re)) .* pagemtimes(Lcol, lap);

            K_adv_diff = Kvisc + Kadv + Ksadv + Ksdif;
            Kuu = (M0 + Msupg + dt*theta*K_adv_diff) .* dV;
            Kup = (dt*(-1)) .* pagemtimes(dNxC, Nt) .* dV;
            Kvp = (dt*(-1)) .* pagemtimes(dNyC, Nt) .* dV;

            Mpu_x = tau_pspg .* pagemtimes(dNxC, Nt);
            Mpu_y = tau_pspg .* pagemtimes(dNyC, Nt);
            Dx    = pagemtimes(N, dNx(1,:,:));
            Dy    = pagemtimes(N, dNx(2,:,:));
            Dpx   = tau_pspg .* pagemtimes(dNxC, Ladv);
            Dpy   = tau_pspg .* pagemtimes(dNyC, Ladv);
            Pvx   = -(tau_pspg/Re) .* pagemtimes(dNxC, lap);
            Pvy   = -(tau_pspg/Re) .* pagemtimes(dNyC, lap);

            Kpu = (Mpu_x + dt*Dx + dt*theta*(Dpx + Pvx)) .* dV;
            Kpv = (Mpu_y + dt*Dy + dt*theta*(Dpy + Pvy)) .* dV;
            Kpp = (dt .* tau_pspg) .* pagemtimes(permute(dNx,[2 1 3]), dNx) .* dV;

            Ke(iu,iu,:) = Ke(iu,iu,:) + Kuu;
            Ke(iv,iv,:) = Ke(iv,iv,:) + Kuu;
            Ke(iu,ip,:) = Ke(iu,ip,:) + Kup;
            Ke(iv,ip,:) = Ke(iv,ip,:) + Kvp;
            Ke(ip,iu,:) = Ke(ip,iu,:) + Kpu;
            Ke(ip,iv,:) = Ke(ip,iv,:) + Kpv;
            Ke(ip,ip,:) = Ke(ip,ip,:) + Kpp;

            un_reshaped = reshape(Un_l(1:3:end,:), nen, 1, ne);
            vn_reshaped = reshape(Un_l(2:3:end,:), nen, 1, ne);
            
            explicit_u = pagemtimes(K_adv_diff, un_reshaped) .* (-dt * (1-theta));
            explicit_v = pagemtimes(K_adv_diff, vn_reshaped) .* (-dt * (1-theta));
            
            explicit_p = pagemtimes(Dpx + Pvx, un_reshaped) + pagemtimes(Dpy + Pvy, vn_reshaped);
            explicit_p = explicit_p .* (-dt * (1-theta));

            NpL  = N + tau_supg .* Lcol;
            Feu  = (NpL .* (dt*fx + ugn) + explicit_u) .* dV;
            Fev  = (NpL .* (dt*fy + vgn) + explicit_v) .* dV;
            Fep  = (tau_pspg .* (dt*(dNxC.*fx + dNyC.*fy) + (dNxC.*ugn + dNyC.*vgn)) + explicit_p) .* dV;

            Fe(iu,1,:) = Fe(iu,1,:) + Feu;
            Fe(iv,1,:) = Fe(iv,1,:) + Fev;
            Fe(ip,1,:) = Fe(ip,1,:) + Fep;
        end

        r0 = repmat((1:ndof_e)', 1, ndof_e);
        c0 = repmat(1:ndof_e,  ndof_e, 1);
        r0 = r0(:);  c0 = c0(:);

        I_t = gather(edofs_g(:, r0)');
        J_t = gather(edofs_g(:, c0)');
        V_t = gather(reshape(Ke, ndof_e^2, ne));

        I_all = [I_all; I_t(:)];
        J_all = [J_all; J_t(:)];
        V_all = [V_all; V_t(:)];

        ed_cpu = gather(edofs_g');
        fe_cpu = gather(reshape(Fe, ndof_e, ne));
        F_cpu  = F_cpu + accumarray(ed_cpu(:), fe_cpu(:), [total_dof, 1]);
    end

    I = I_all;
    J = J_all;
    V = V_all;
    F = F_cpu;
end