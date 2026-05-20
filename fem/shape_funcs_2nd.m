function d2N = shape_funcs_2nd(xi, eta, type)
    % SHAPE_FUNCS_2ND Returns the local second derivatives for quadratic elements.
    % Output d2N is a 3 x NEN matrix: [d2N/dxi^2; d2N/deta^2; d2N/dxi_deta]
    
    switch type
        case 'tri6'
            % Exact analytical derivatives [cite: 127-133]
            d2N = [ 4,  4,  0, -8,  0,  0;
                    4,  0,  4,  0,  0, -8;
                    4,  0,  0, -4,  4, -4];
                    
        case 'quad9'
            d2N = zeros(3, 9);
            
            % d2N / dxi^2
            d2N(1,1) = 0.5*(eta^2-eta); d2N(1,2) = 0.5*(eta^2-eta); 
            d2N(1,3) = 0.5*(eta^2+eta); d2N(1,4) = 0.5*(eta^2+eta);
            d2N(1,5) = -(eta^2-eta);    d2N(1,6) = (1-eta^2);       % FIXED
            d2N(1,7) = -(eta^2+eta);    d2N(1,8) = (1-eta^2);       % FIXED
            d2N(1,9) = -2*(1-eta^2); 
            
            % d2N / deta^2
            d2N(2,1) = 0.5*(xi^2-xi);   d2N(2,2) = 0.5*(xi^2+xi);   
            d2N(2,3) = 0.5*(xi^2+xi);   d2N(2,4) = 0.5*(xi^2-xi);
            d2N(2,5) = (1-xi^2);        d2N(2,6) = -(xi^2+xi);      % FIXED
            d2N(2,7) = (1-xi^2);        d2N(2,8) = -(xi^2-xi);      % FIXED
            d2N(2,9) = -2*(1-xi^2); 
            
            % d2N / dxi_deta (These were already correct)
            d2N(3,1) = 0.25*(2*xi-1)*(2*eta-1); d2N(3,2) = 0.25*(2*xi+1)*(2*eta-1); 
            d2N(3,3) = 0.25*(2*xi+1)*(2*eta+1); d2N(3,4) = 0.25*(2*xi-1)*(2*eta+1);
            d2N(3,5) = -xi*(2*eta-1);           d2N(3,6) = -eta*(2*xi+1);           
            d2N(3,7) = -xi*(2*eta+1);           d2N(3,8) = -eta*(2*xi-1);
            d2N(3,9) = 4*xi*eta;
    end
end