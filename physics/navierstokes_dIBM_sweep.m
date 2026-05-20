function [I, J, V, F] = navierstokes_dIBM_sweep(mesh, conn, edof, dof, Re, dt, U_k, U_n, f_source, c_eps, c_k)
% NAVIERSTOKES_DIBM  GPU-vectorised assembly of the transient 2D Navier-Stokes
% with Diffuse Immersed Boundary Method (Brinkman Volume Penalization).
%
% IDENTICAL to navierstokes.m except for the Brinkman volume penalization
% (diffuse IBM). The penalty Psi = chi/(Re*K_p) is a reaction operator.
%
% It is tested by Galerkin (N), SUPG (tau*u·grad(N)), and PSPG (tau*grad(q))
% for full stabilization consistency.

    total_dof = dof.ndof;
    ndpn = 3;
    F_cpu = zeros(total_dof, 1);

    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    I_all = []; J_all = []; V_all = [];

    U_k_g = gpuArray(U_k(:));
    U_n_g = gpuArray(U_n(:));
    nodes_g = gpuArray(mesh.nodes(:,1:2));

    % --- IBM parameters ---
    D_cyl = 0.1;  R_cyl = D_cyl / 2;
    xc = 0.2;     yc = 0.2;
    epsilon = 0.0005;    % sharp interface (sub-element)
    K_p     = 1e-5;     % Darcy number — Psi_max = 1/(Re*K_p)

    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isfield(conn, type) || isempty(conn.(type)), continue; end

        elems  = conn.(type);
        edofs  = edof.(type);
        ne     = size(elems, 1);
        nen    = size(elems, 2);
        ndof_e = nen * ndpn;

        % FIX 1: Use SAME quadrature as standard navierstokes.m
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

        uk = Uk_l(1:3:end, :);   vk = Uk_l(2:3:end, :);
        un = Un_l(1:3:end, :);   vn = Un_l(2:3:end, :);

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

        um = reshape(mean(uk,1), 1,1,ne);
        vm = reshape(mean(vk,1), 1,1,ne);
        umag = sqrt(um.^2 + vm.^2);
        % Base tau components (before IBM reaction — finalised per Gauss pt)
        tau_base_inv2 = (2/dt)^2 + (2*umag./h_e).^2 + (4./(Re*h_e.^2)).^2;

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

            ugk = reshape(N'*uk, 1,1,ne);
            vgk = reshape(N'*vk, 1,1,ne);
            ugn = reshape(N'*un, 1,1,ne);
            vgn = reshape(N'*vn, 1,1,ne);

            % === Brinkman IBM indicator dynamically scaled onto mesh density ===
            eps_e = c_eps * h_e;             % Interface dynamically tracks exactly 1 mesh-width
            Kp_e  = c_k * (h_e.^2);       % Darcy permeability naturally scales with element area to prevent solver singularity
            %disp(max(eps_e))
            %disp(max(Kp_e))
            dq    = sqrt((xg - xc).^2 + (yg - yc).^2) - R_cyl;
            chi_q = 0.5 * (1 - tanh(dq ./ eps_e));
            Psi_q = chi_q ./ (Re * Kp_e);

            % FIX 3: Single tau_s including IBM reaction (advection-diffusion-reaction)
            % In fluid (Psi≈0): tau_s unchanged from standard.
            % In solid (Psi≈25000): tau_s ≈ 1/Psi → tau_s*Psi = O(1).
            tau_s = 1 ./ sqrt(tau_base_inv2 + Psi_q.^2);

            % FIX 2: Single computation of advection operator (no duplicate)
            Ladv = ugk.*dNx(1,:,:) + vgk.*dNx(2,:,:);

            Lcol  = permute(Ladv, [2 1 3]);
            Nt    = reshape(N', 1, nen);
            dNxC  = permute(dNx(1,:,:), [2 1 3]);
            dNyC  = permute(dNx(2,:,:), [2 1 3]);

            M0    = N * Nt;
            Msupg = tau_s .* pagemtimes(Lcol, Nt);
            Kvisc = (1/Re) * pagemtimes(permute(dNx,[2 1 3]), dNx);
            Kadv  = pagemtimes(N, Ladv);
            Ksadv = tau_s .* pagemtimes(Lcol, Ladv);
            Ksdif = tau_s .* (-(1/Re)) .* pagemtimes(Lcol, lap);

            % Galerkin AND SUPG consistent IBM reaction:
            % The penalty must be tested against both N and tau*(u · ∇N)
            Kibm_u = Psi_q .* (M0 + Msupg); 

            Kuu = (M0 + Msupg + dt*(Kvisc + Kadv + Ksadv + Ksdif + Kibm_u)) .* dV;

            Kup = (dt*(-1)) .* pagemtimes(dNxC, Nt) .* dV;
            Kvp = (dt*(-1)) .* pagemtimes(dNyC, Nt) .* dV;

            Mpu_x = tau_s .* pagemtimes(dNxC, Nt);
            Mpu_y = tau_s .* pagemtimes(dNyC, Nt);
            Dx    = pagemtimes(N, dNx(1,:,:));
            Dy    = pagemtimes(N, dNx(2,:,:));
            Dpx   = tau_s .* pagemtimes(dNxC, Ladv);
            Dpy   = tau_s .* pagemtimes(dNyC, Ladv);
            Pvx   = -(tau_s/Re) .* pagemtimes(dNxC, lap);
            Pvy   = -(tau_s/Re) .* pagemtimes(dNyC, lap);
            
            % PSPG consistent IBM reaction: psi*u tested against tau*∇q
            Kpu_ibm = Psi_q .* Mpu_x;
            Kpv_ibm = Psi_q .* Mpu_y;

            Kpu = (Mpu_x + dt*(Dx + Dpx + Pvx + Kpu_ibm)) .* dV;
            Kpv = (Mpu_y + dt*(Dy + Dpy + Pvy + Kpv_ibm)) .* dV;
            Kpp = (dt .* tau_s) .* pagemtimes(permute(dNx,[2 1 3]), dNx) .* dV;

            Ke(iu,iu,:) = Ke(iu,iu,:) + Kuu;
            Ke(iv,iv,:) = Ke(iv,iv,:) + Kuu;
            Ke(iu,ip,:) = Ke(iu,ip,:) + Kup;
            Ke(iv,ip,:) = Ke(iv,ip,:) + Kvp;
            Ke(ip,iu,:) = Ke(ip,iu,:) + Kpu;
            Ke(ip,iv,:) = Ke(ip,iv,:) + Kpv;
            Ke(ip,ip,:) = Ke(ip,ip,:) + Kpp;

            NpL  = N + tau_s .* Lcol;
            Feu  = NpL .* (dt*fx + ugn) .* dV;
            Fev  = NpL .* (dt*fy + vgn) .* dV;
            Fep  = tau_s .* (dt*(dNxC.*fx + dNyC.*fy) ...
                           + (dNxC.*ugn + dNyC.*vgn)) .* dV;

            % NOTE: IBM penalty has ZERO RHS contribution because u_desired=0.
            % The penalty Ψ·u is purely implicit (acts only on u^{n+1}).

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
