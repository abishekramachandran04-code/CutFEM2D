function [W, xi, eta] = cut_quadrature(subtriangles, elem_nodes, elem_type, n_gauss)
% CUT_QUADRATURE Generates mapped Gauss points and weights for cut elements.
%   [W, xi, eta] = cut_quadrature(subtriangles, elem_nodes, elem_type, n_gauss)
%   Maps points from standard sub-triangles back to the parent reference domain.

    if nargin < 4
        n_gauss = 3;
    end
    
    [xi_t, eta_t, w_t] = get_triangle_gauss_points(n_gauss);
    
    num_subs = size(subtriangles, 1);
    num_pts = num_subs * n_gauss;
    
    W = zeros(num_pts, 1);
    xi = zeros(num_pts, 1);
    eta = zeros(num_pts, 1);
    
    if num_subs == 0
        return;
    end
    
    idx = 1;
    for s = 1:num_subs
        p1 = subtriangles(s, 1:2);
        p2 = subtriangles(s, 3:4);
        p3 = subtriangles(s, 5:6);
        
        J_tri = [p2(1) - p1(1), p3(1) - p1(1);
                 p2(2) - p1(2), p3(2) - p1(2)];
        detJ_tri = abs(det(J_tri));
        
        for g = 1:n_gauss
            x_g = p1(1) + (p2(1) - p1(1))*xi_t(g) + (p3(1) - p1(1))*eta_t(g);
            y_g = p1(2) + (p2(2) - p1(2))*xi_t(g) + (p3(2) - p1(2))*eta_t(g);
            
            % detJ_tri scales the standard reference triangle (area 0.5) to physical
            % w_t sum to 0.5 natively, so detJ_tri directly scales it.
            W(idx) = w_t(g) * detJ_tri; 
            
            [xi_p, eta_p] = inverse_mapping(x_g, y_g, elem_nodes, elem_type);
            xi(idx) = xi_p;
            eta(idx) = eta_p;
            idx = idx + 1;
        end
    end
end

function [xi_t, eta_t, w_t] = get_triangle_gauss_points(n_gauss)
    if n_gauss == 1
        xi_t = 1/3;
        eta_t = 1/3;
        w_t = 1/2;
    elseif n_gauss == 3
        xi_t = [1/6; 2/3; 1/6];
        eta_t = [1/6; 1/6; 2/3];
        w_t = [1/6; 1/6; 1/6];
    elseif n_gauss == 7
        xi_t = [1/3; 0.470142064105115; 0.470142064105115; 0.059715871789770; ...
                0.101286507323456; 0.101286507323456; 0.797426985353087];
        eta_t = [1/3; 0.059715871789770; 0.470142064105115; 0.470142064105115; ...
                 0.101286507323456; 0.797426985353087; 0.101286507323456];
        w_t = [0.1125; 0.066197076394253; 0.066197076394253; 0.066197076394253; ...
               0.062969590272413; 0.062969590272413; 0.062969590272413];
    else
        error('Unsupported number of Gauss points for triangle.');
    end
end

function [xi, eta] = inverse_mapping(x, y, elem_nodes, elem_type)
    if strcmp(elem_type, 'quad4') || strcmp(elem_type, 'quad9')
        xi = 0.0; eta = 0.0;
    else
        xi = 1/3; eta = 1/3;
    end
    
    max_iter = 20;
    tol = 1e-10;
    
    for iter = 1:max_iter
        [N, dN_dxi] = shape_funcs(xi, eta, elem_type);
        
        x_curr = dot(N, elem_nodes(:, 1));
        y_curr = dot(N, elem_nodes(:, 2));
        
        R = [x - x_curr; y - y_curr];
        if norm(R) < tol
            break;
        end
        
        dx_dxi = dot(dN_dxi(1,:), elem_nodes(:, 1));
        dy_dxi = dot(dN_dxi(1,:), elem_nodes(:, 2));
        dx_deta = dot(dN_dxi(2,:), elem_nodes(:, 1));
        dy_deta = dot(dN_dxi(2,:), elem_nodes(:, 2));
        
        J = [dx_dxi, dx_deta;
             dy_dxi, dy_deta];
             
        delta = J \ R;
        xi = xi + delta(1);
        eta = eta + delta(2);
    end
end
