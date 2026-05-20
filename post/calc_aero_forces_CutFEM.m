function [F_D, F_L, C_D, C_L] = calc_aero_forces_CutFEM(mesh, conn, edof, dof, Re, U, phi, classification, cutfem_params, D, U_inf)
% CALC_AERO_FORCES_CUTFEM  Computes aerodynamic drag & lift for CutFEM
% via the variationally consistent Nitsche flux along the immersed interface.
%
% The Nitsche multiplier (boundary traction on the body) is:
%   lambda = -sigma·n + gamma/(Re·h) * (u - u_D)
%
% where n is the outward normal of the fluid domain (points INTO the solid),
% sigma = -pI + (1/Re)*grad(u), and u_D = 0 (no-slip).
%
% Force on body:  F = integral_Gamma lambda dGamma
%   F_D = integral [ p*n_x - (1/Re)*(grad u · n)_x + gamma/(Re*h)*u ] dGamma
%   F_L = integral [ p*n_y - (1/Re)*(grad u · n)_y + gamma/(Re*h)*v ] dGamma

    F_D = 0;
    F_L = 0;
    
    U_cpu = gather(U(:));
    nodes_cpu = mesh.nodes(:,1:2);
    gamma_u = cutfem_params.gamma_u;
    
    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    
    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isfield(classification, type), continue; end
        
        cut_idx = classification.(type).CUT;
        if isempty(cut_idx), continue; end
        
        elems = conn.(type);
        edofs = edof.(type);
        nen = size(elems, 2);
        ndof_e = nen * 3;
        iu = 1:3:ndof_e;
        iv = 2:3:ndof_e;
        ip = 3:3:ndof_e;
        
        for i = 1:length(cut_idx)
            e_global = cut_idx(i);
            nodes_e = elems(e_global, :);
            edofs_e = edofs(e_global, :);
            coords_e = nodes_cpu(nodes_e, :);
            phi_e = phi(nodes_e);
            
            Ue = U_cpu(edofs_e);
            uk_e = Ue(iu);
            vk_e = Ue(iv);
            pk_e = Ue(ip);
            
            % Compute element size h_e (same formula as navierstokes_cutfem)
            if contains(type, 'tri'), nc = 3; else, nc = 4; end
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
            if h_e < 1e-12, h_e = 1e-12; end
            
            % Nitsche line quadrature along the zero level-set
            [W_line, xi_line, eta_line, nx_q, ny_q] = nitsche_line_quadrature(coords_e, phi_e, type, 3);
            
            for q = 1:length(W_line)
                [N, dNxi] = shape_funcs(xi_line(q), eta_line(q), type);
                Jm = dNxi * coords_e;
                dJ = Jm(1,1)*Jm(2,2) - Jm(1,2)*Jm(2,1);
                iJ = [Jm(2,2), -Jm(1,2); -Jm(2,1), Jm(1,1)] / dJ;
                dNx = iJ * dNxi;
                dL = W_line(q);
                
                n_x = nx_q(q);
                n_y = ny_q(q);
                
                % Interpolate solution at the Gauss point
                u_g = dot(N, uk_e);
                v_g = dot(N, vk_e);
                p_g = dot(N, pk_e);
                
                % Velocity gradients
                du_dx = dot(dNx(1,:), uk_e);
                du_dy = dot(dNx(2,:), uk_e);
                dv_dx = dot(dNx(1,:), vk_e);
                dv_dy = dot(dNx(2,:), vk_e);
                
                % Cauchy traction  sigma·n  (n = outward from fluid)
                sigma_n_x = -p_g * n_x + (1/Re) * (du_dx * n_x + du_dy * n_y);
                sigma_n_y = -p_g * n_y + (1/Re) * (dv_dx * n_x + dv_dy * n_y);
                
                % Fetch instantaneous cylinder velocity
                if isfield(cutfem_params, 'uD'), uD = cutfem_params.uD; else, uD = 0; end
                if isfield(cutfem_params, 'vD'), vD = cutfem_params.vD; else, vD = 0; end
                
                % Nitsche penalty correction for moving boundary
                pen_x = (gamma_u / (Re * h_e)) * (u_g - uD);
                pen_y = (gamma_u / (Re * h_e)) * (v_g - vD);
                
                % Variationally consistent Nitsche flux:
                %   lambda = -sigma·n + gamma/(Re*h) * u
                % Force on body = integral lambda dGamma
                F_D = F_D + (-sigma_n_x + pen_x) * dL;
                F_L = F_L + (-sigma_n_y + pen_y) * dL;
            end
        end
    end
    
    F_D = full(gather(F_D));
    F_L = full(gather(F_L));
    
    % Non-dimensionalise
    q_inf = 0.5 * 1.0 * U_inf^2;   % rho = 1
    C_D = F_D / (q_inf * D);
    C_L = F_L / (q_inf * D);
end
