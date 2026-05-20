function subtriangles = subtriangulate_cut_element(elem_nodes, phi_nodes, elem_type)
% SUBTRIANGULATE_CUT_ELEMENT Slices the fluid polygon of a cut element.
%   subtriangles = subtriangulate_cut_element(elem_nodes, phi_nodes, elem_type)
%   Fluid is phi <= 0.
%   Returns a [N x 6] matrix where each row is [x1 y1 x2 y2 x3 y3] for a fluid sub-triangle.

    subtriangles = [];
    
    if strcmp(elem_type, 'tri6')
        % Subdivide into 4 tri3s
        % Nodes: 1, 2, 3 (corners), 4 (mid 1-2), 5 (mid 2-3), 6 (mid 3-1)
        sub_conns = [1 4 6; 4 2 5; 6 5 3; 4 5 6];
        for i = 1:4
            sub_nodes = elem_nodes(sub_conns(i,:), :);
            sub_phis = phi_nodes(sub_conns(i,:));
            sub_tris = process_linear_polygon(sub_nodes, sub_phis);
            subtriangles = [subtriangles; sub_tris];
        end
        
    elseif strcmp(elem_type, 'quad9')
        % Subdivide into 4 quad4s
        % Nodes: 1-4 corners, 5-8 mid-edges (1-2, 2-3, 3-4, 4-1), 9 center
        sub_conns = [1 5 9 8; 5 2 6 9; 9 6 3 7; 8 9 7 4];
        for i = 1:4
            sub_nodes = elem_nodes(sub_conns(i,:), :);
            sub_phis = phi_nodes(sub_conns(i,:));
            sub_tris = process_linear_polygon(sub_nodes, sub_phis);
            subtriangles = [subtriangles; sub_tris];
        end
        
    elseif strcmp(elem_type, 'tri3') || strcmp(elem_type, 'quad4')
        subtriangles = process_linear_polygon(elem_nodes, phi_nodes);
    end
end

function subtriangles = process_linear_polygon(nodes, phis)
    n = length(phis);
    
    if all(phis <= 0)
        % Entirely fluid
        if n == 3
            subtriangles = [nodes(1,1:2), nodes(2,1:2), nodes(3,1:2)];
        else % n == 4
            subtriangles = [nodes(1,1:2), nodes(2,1:2), nodes(3,1:2);
                            nodes(1,1:2), nodes(3,1:2), nodes(4,1:2)];
        end
        return;
    end
    
    if all(phis > 0)
        % Entirely solid
        subtriangles = [];
        return;
    end
    
    fluid_poly_nodes = [];
    
    for i = 1:n
        j = mod(i, n) + 1;
        
        if phis(i) <= 0
            fluid_poly_nodes = [fluid_poly_nodes; nodes(i, :)];
        end
        
        if phis(i) * phis(j) < 0
            % Compute intersection linearly
            t = phis(i) / (phis(i) - phis(j));
            p_int = nodes(i, :) + t * (nodes(j, :) - nodes(i, :));
            fluid_poly_nodes = [fluid_poly_nodes; p_int];
        end
    end
    
    if isempty(fluid_poly_nodes) || size(fluid_poly_nodes, 1) < 3
        subtriangles = [];
        return;
    end
    
    % Sort polygon vertices counter-clockwise
    center = mean(fluid_poly_nodes, 1);
    angles = atan2(fluid_poly_nodes(:,2) - center(2), fluid_poly_nodes(:,1) - center(1));
    [~, sort_idx] = sort(angles);
    sorted_nodes = fluid_poly_nodes(sort_idx, :);
    
    % Triangulate the polygon
    m = size(sorted_nodes, 1);
    subtriangles = zeros(m - 2, 6);
    p1 = sorted_nodes(1, :);
    for k = 2:m-1
        p2 = sorted_nodes(k, :);
        p3 = sorted_nodes(k+1, :);
        subtriangles(k-1, :) = [p1(1:2), p2(1:2), p3(1:2)];
    end
end
