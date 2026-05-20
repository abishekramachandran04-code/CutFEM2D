function mesh = fix_orientation(mesh)
    % FIX_ORIENTATION Ensures all 2D elements have counter-clockwise node ordering.
    
    elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
    
    for t = 1:length(elem_types)
        type = elem_types{t};
        if isfield(mesh, type) && ~isempty(mesh.(type))
            elems = mesh.(type);
            n = mesh.nodes;
            
            % Extract corner nodes to check orientation
            if contains(type, 'tri')
                corners = elems(:, 1:3);
            else % quad
                corners = elems(:, 1:4);
            end
            
            % Extract 1D arrays first to prevent MATLAB from flattening
            X_all = n(:, 1);
            Y_all = n(:, 2);
            
            % Map the corners
            x = X_all(corners);
            y = Y_all(corners);
            
            % Compute signed area using the Shoelace formula
            if contains(type, 'tri')
                signed_area = 0.5 * ((x(:,1).*y(:,2) - x(:,2).*y(:,1)) + ...
                                     (x(:,2).*y(:,3) - x(:,3).*y(:,2)) + ...
                                     (x(:,3).*y(:,1) - x(:,1).*y(:,3)));
            else % quad
                signed_area = 0.5 * ((x(:,1).*y(:,2) - x(:,2).*y(:,1)) + ...
                                     (x(:,2).*y(:,3) - x(:,3).*y(:,2)) + ...
                                     (x(:,3).*y(:,4) - x(:,4).*y(:,3)) + ...
                                     (x(:,4).*y(:,1) - x(:,1).*y(:,4)));
            end
            
            % Find elements that are clockwise (negative area)
            bad_idx = find(signed_area < 0);
            
            if ~isempty(bad_idx)
                % Correct the node ordering based on element type
                switch type
                    case 'tri3'
                        mesh.tri3(bad_idx, [2, 3]) = mesh.tri3(bad_idx, [3, 2]);
                    case 'quad4'
                        mesh.quad4(bad_idx, [2, 4]) = mesh.quad4(bad_idx, [4, 2]);
                    case 'tri6'
                        mesh.tri6(bad_idx, [2, 3, 4, 6]) = mesh.tri6(bad_idx, [3, 2, 6, 4]);
                    case 'quad9'
                        mesh.quad9(bad_idx, [2, 4, 5, 6, 7, 8]) = mesh.quad9(bad_idx, [4, 2, 8, 7, 6, 5]);
                end
                fprintf('Corrected CCW orientation for %d inverted %s elements.\n', length(bad_idx), type);
            end
        end
    end
end