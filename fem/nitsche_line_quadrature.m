function [W, xi, eta, nx, ny] = nitsche_line_quadrature(elem_nodes, phi_nodes, elem_type, n_gauss_1d)
% NITSCHE_LINE_QUADRATURE Generates 1D Gauss points for the immersed boundary.
%   [W, xi, eta, nx, ny] = nitsche_line_quadrature(elem_nodes, phi_nodes, elem_type, n_gauss_1d)

    if nargin < 4
        n_gauss_1d = 3;
    end
    
    line_segments = get_cut_segments(elem_nodes, phi_nodes, elem_type);
    
    [xi_1d, w_1d] = get_1d_gauss_points(n_gauss_1d);
    
    num_segs = size(line_segments, 1);
    num_pts = num_segs * n_gauss_1d;
    
    W = zeros(num_pts, 1);
    xi = zeros(num_pts, 1);
    eta = zeros(num_pts, 1);
    nx = zeros(num_pts, 1);
    ny = zeros(num_pts, 1);
    
    if num_segs == 0
        return;
    end
    
    idx = 1;
    for s = 1:num_segs
        p1 = line_segments(s, 1:2);
        p2 = line_segments(s, 3:4);
        
        dx = p2(1) - p1(1);
        dy = p2(2) - p1(2);
        L = sqrt(dx^2 + dy^2);
        
        % Base normal (dy, -dx)
        n_x = dy;
        n_y = -dx;
        n_len = sqrt(n_x^2 + n_y^2);
        n_x = n_x / n_len;
        n_y = n_y / n_len;
        
        % Ensure normal points out of fluid (towards solid, phi > 0)
        mid = (p1 + p2) / 2;
        phi_test = evaluate_interpolated_phi(mid(1) + 1e-4*n_x, mid(2) + 1e-4*n_y, elem_nodes, elem_type, phi_nodes);
        phi_mid = evaluate_interpolated_phi(mid(1), mid(2), elem_nodes, elem_type, phi_nodes);
        
        if phi_test < phi_mid
            n_x = -n_x;
            n_y = -n_y;
        end
        
        for g = 1:n_gauss_1d
            % Map xi_1d in [-1, 1] to t in [0, 1]
            t = 0.5 * (1 + xi_1d(g));
            x_g = p1(1) + t * dx;
            y_g = p1(2) + t * dy;
            
            % 1D Gauss weights sum to 2 over [-1, 1]. Length mapping introduces factor L/2.
            W(idx) = (L / 2) * w_1d(g);
            nx(idx) = n_x;
            ny(idx) = n_y;
            
            [xi_p, eta_p] = inverse_mapping(x_g, y_g, elem_nodes, elem_type);
            xi(idx) = xi_p;
            eta(idx) = eta_p;
            
            idx = idx + 1;
        end
    end
end

function segments = get_cut_segments(elem_nodes, phi_nodes, elem_type)
    segments = [];
    
    if strcmp(elem_type, 'tri6')
        sub_conns = [1 4 6; 4 2 5; 6 5 3; 4 5 6];
        for i = 1:4
            sub_nodes = elem_nodes(sub_conns(i,:), :);
            sub_phis = phi_nodes(sub_conns(i,:));
            seg = process_linear_segment(sub_nodes, sub_phis);
            if ~isempty(seg)
                segments = [segments; seg];
            end
        end
    elseif strcmp(elem_type, 'quad9')
        sub_conns = [1 5 9 8; 5 2 6 9; 9 6 3 7; 8 9 7 4];
        for i = 1:4
            sub_nodes = elem_nodes(sub_conns(i,:), :);
            sub_phis = phi_nodes(sub_conns(i,:));
            seg = process_linear_segment(sub_nodes, sub_phis);
            if ~isempty(seg)
                segments = [segments; seg];
            end
        end
    elseif strcmp(elem_type, 'tri3') || strcmp(elem_type, 'quad4')
        segments = process_linear_segment(elem_nodes, phi_nodes);
    end
end

function seg = process_linear_segment(nodes, phis)
    n = length(phis);
    pts = [];
    for i = 1:n
        j = mod(i, n) + 1;
        if phis(i) * phis(j) < 0
            t = phis(i) / (phis(i) - phis(j));
            p_int = nodes(i, :) + t * (nodes(j, :) - nodes(i, :));
            pts = [pts; p_int];
        end
    end
    if size(pts, 1) == 2
        seg = [pts(1, 1:2), pts(2, 1:2)];
    else
        seg = [];
    end
end

function phi_val = evaluate_interpolated_phi(x, y, elem_nodes, elem_type, phi_nodes)
    [xi, eta] = inverse_mapping(x, y, elem_nodes, elem_type);
    [N, ~] = shape_funcs(xi, eta, elem_type);
    phi_val = dot(N, phi_nodes);
end

function [xi_1d, w_1d] = get_1d_gauss_points(n)
    if n == 1
        xi_1d = 0; w_1d = 2;
    elseif n == 2
        xi_1d = [-1/sqrt(3); 1/sqrt(3)];
        w_1d = [1; 1];
    elseif n == 3
        xi_1d = [-sqrt(3/5); 0; sqrt(3/5)];
        w_1d = [5/9; 8/9; 5/9];
    else
        error('Unsupported 1D Gauss points.');
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
        J = [dx_dxi, dx_deta; dy_dxi, dy_deta];
        delta = J \ R;
        xi = xi + delta(1);
        eta = eta + delta(2);
    end
end
