function [N, dN_dxi] = shape_funcs(xi, eta, elem_type)
    % N: [n_nodes x 1] vector of shape function values
    % dN_dxi: [2 x n_nodes] matrix of local derivatives
    %         row 1: dN/dxi, row 2: dN/deta
    
    switch lower(elem_type)
        case 'tri3'
            N = [1 - xi - eta; 
                 xi; 
                 eta];
                 
            dN_dxi = [-1, 1, 0;
                      -1, 0, 1];
                      
        case 'tri6'
            L1 = 1 - xi - eta; 
            L2 = xi; 
            L3 = eta;
            
            N = [L1 * (2*L1 - 1);
                 L2 * (2*L2 - 1);
                 L3 * (2*L3 - 1);
                 4 * L1 * L2;
                 4 * L2 * L3;
                 4 * L3 * L1];
                 
            dN_dxi = [4*xi + 4*eta - 3,  4*xi - 1,  0,          4 - 8*xi - 4*eta,  4*eta, -4*eta;
                      4*xi + 4*eta - 3,  0,         4*eta - 1, -4*xi,              4*xi,   4 - 4*xi - 8*eta];
                      
        case 'quad4'
            N = 0.25 * [(1-xi)*(1-eta);
                        (1+xi)*(1-eta);
                        (1+xi)*(1+eta);
                        (1-xi)*(1+eta)];
                        
            dN_dxi = 0.25 * [-(1-eta),  (1-eta), (1+eta), -(1+eta);
                             -(1-xi),  -(1+xi),  (1+xi),   (1-xi)];
                             
        case 'quad9'
            % 1D shape functions and derivatives
            N1_xi = 0.5*xi*(xi-1);   dN1_xi = xi - 0.5;
            N2_xi = 1 - xi^2;        dN2_xi = -2*xi;
            N3_xi = 0.5*xi*(xi+1);   dN3_xi = xi + 0.5;
            
            N1_et = 0.5*eta*(eta-1); dN1_et = eta - 0.5;
            N2_et = 1 - eta^2;       dN2_et = -2*eta;
            N3_et = 0.5*eta*(eta+1); dN3_et = eta + 0.5;
            
            N = [N1_xi * N1_et;
                 N3_xi * N1_et;
                 N3_xi * N3_et;
                 N1_xi * N3_et;
                 N2_xi * N1_et;
                 N3_xi * N2_et;
                 N2_xi * N3_et;
                 N1_xi * N2_et;
                 N2_xi * N2_et];
                 
            dN_dxi = [dN1_xi*N1_et, dN3_xi*N1_et, dN3_xi*N3_et, dN1_xi*N3_et, dN2_xi*N1_et, dN3_xi*N2_et, dN2_xi*N3_et, dN1_xi*N2_et, dN2_xi*N2_et;
                      N1_xi*dN1_et, N3_xi*dN1_et, N3_xi*dN3_et, N1_xi*dN3_et, N2_xi*dN1_et, N3_xi*dN2_et, N2_xi*dN3_et, N1_xi*dN2_et, N2_xi*dN2_et];
                      
        otherwise
            error('Unsupported element type: %s', elem_type);
    end
end