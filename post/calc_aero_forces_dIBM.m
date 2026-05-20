function [F_D, F_L, C_D, C_L] = calc_aero_forces_dIBM(mesh, conn, edof, dof, Re, U, D, U_inf)
% CALC_AERO_FORCES_DIBM  Computes aerodynamic forces via volume integral
% of the Brinkman penalty forcing: F = integral( chi/(Re*K_p) * u dV ).
%
% Parameters MUST match navierstokes_dIBM.m exactly.

    F_D = 0; 
    F_L = 0;
    
    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    U_g = U(:);
    
    % --- Must match navierstokes_dIBM.m ---
    xc = 0.2;  yc = 0.2;  R_cyl = D / 2;
    
    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isfield(conn, type) || isempty(conn.(type)), continue; end
        
        elems = conn.(type); 
        edofs = edof.(type);
        ne = size(elems, 1); 
        nen = size(elems, 2);
        
        [xi, eta, wt] = gauss_quadrature(type);
        
        idx_f = reshape(elems', [], 1);
        crd_f = mesh.nodes(idx_f, 1:2);
        coords = permute(reshape(crd_f, nen, ne, 2), [1 3 2]);
        
        ed_f = reshape(edofs', [], 1);
        U_l = reshape(U_g(ed_f), nen*3, ne);
        uk = U_l(1:3:end, :); 
        vk = U_l(2:3:end, :);

        Ae = zeros(1, 1, ne);
        for q = 1:length(wt)
            [~, dNq] = shape_funcs(xi(q), eta(q), type);
            Jq = pagemtimes(dNq, coords);
            dJq = Jq(1,1,:).*Jq(2,2,:) - Jq(1,2,:).*Jq(2,1,:);
            Ae = Ae + dJq * wt(q);
        end
        
        if contains(type,'tri'), nc=3; else, nc=4; end
        corn = coords(1:nc, :, :);
        nxt  = [corn(2:end,:,:); corn(1,:,:)];
        P_e  = sum(sqrt(sum((nxt - corn).^2, 2)), 1);
        h_e  = 4*Ae ./ P_e;

        for q = 1:length(wt)
            [N, dNxi] = shape_funcs(xi(q), eta(q), type);
            Jm = pagemtimes(dNxi, coords);
            dJ = Jm(1,1,:).*Jm(2,2,:) - Jm(1,2,:).*Jm(2,1,:);
            dV = dJ * wt(q);
            
            xg = sum(N .* coords(:,1,:), 1);
            yg = sum(N .* coords(:,2,:), 1);
            
            eps_e = 1.0 * h_e;
            Kp_e  = 1e-1 * (h_e.^2);
            
            dq = sqrt((xg - xc).^2 + (yg - yc).^2) - R_cyl;
            chi_q = 0.5 * (1 - tanh(dq ./ eps_e));
            Psi_q = chi_q ./ (Re * Kp_e);
            
            % Velocity at Gauss point (column vector dotted with nodal values)
            uq = reshape(N' * uk, 1, 1, ne);
            vq = reshape(N' * vk, 1, 1, ne);
            
            F_D = F_D + sum(Psi_q .* uq .* dV, 'all');
            F_L = F_L + sum(Psi_q .* vq .* dV, 'all');
        end
    end
    F_D = full(gather(F_D));
    F_L = full(gather(F_L));
    
    q_inf = 0.5 * 1.0 * U_inf^2;
    C_D = F_D / (q_inf * D);
    C_L = F_L / (q_inf * D);
end
