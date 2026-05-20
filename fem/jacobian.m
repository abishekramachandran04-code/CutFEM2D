function [dN_dx, detJ] = jacobian(dN_dxi, node_coords)
    % node_coords: [n_nodes x 2] matrix of global coordinates for the element
    % dN_dx: [2 x n_nodes] matrix of global derivatives (row 1: d/dx, row 2: d/dy)
    % detJ: scalar determinant of the Jacobian
    
    % Compute Jacobian matrix: J = [dx/dxi, dy/dxi; dx/deta, dy/deta]
    J = dN_dxi * node_coords;
    
    % Determinant of the Jacobian
    detJ = J(1,1)*J(2,2) - J(1,2)*J(2,1);
    
    % Trap inverted elements
    if detJ <= 0
        error('Negative or zero Jacobian determinant detected. Check mesh connectivity/node ordering.');
    end
    
    % Inverse of 2x2 Jacobian matrix
    invJ = (1/detJ) * [ J(2,2), -J(1,2);
                       -J(2,1),  J(1,1)];
                       
    % Transform local derivatives to global derivatives
    dN_dx = invJ * dN_dxi;
end