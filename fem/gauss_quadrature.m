function [xi, eta, weight] = gauss_quadrature(elem_type)
    % Returns Gauss points and weights for standard 2D elements
    % xi, eta are column vectors of integration points
    % weight is a column vector of integration weights
    
    switch lower(elem_type)
        case 'tri3'
            % 1-point rule (exact for polynomials up to degree 1)
            xi = 1/3; 
            eta = 1/3; 
            weight = 1/2;
            
        case 'tri6'
            % 3-point rule (exact for polynomials up to degree 2)
            xi = [1/6; 2/3; 1/6];
            eta = [1/6; 1/6; 2/3];
            weight = [1/6; 1/6; 1/6];
            
        case 'quad4'
            % 2x2 Gauss-Legendre rule
            pt = 1/sqrt(3);
            xi = [-pt; pt; pt; -pt];
            eta = [-pt; -pt; pt; pt];
            weight = [1; 1; 1; 1];
            
        case 'quad9'
            % 3x3 Gauss-Legendre rule
            pts = [-sqrt(3/5), 0, sqrt(3/5)];
            wts = [5/9, 8/9, 5/9];
            [X, Y] = meshgrid(pts, pts);
            [Wx, Wy] = meshgrid(wts, wts);
            xi = X(:); 
            eta = Y(:);
            weight = Wx(:) .* Wy(:);
            
        otherwise
            error('Unsupported element type: %s', elem_type);
    end
end