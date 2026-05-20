function U = gauss_elim(K, F)
    % GAUSS_ELIM Custom solver using Gaussian Elimination with partial pivoting
    
    n = size(K, 1);
    U = zeros(n, 1);
    
    % Convert to full matrix for the manual loop (prevents severe sparse-indexing slowdowns)
    if issparse(K)
        K = full(K);
    end
    
    % Create augmented matrix [K | F]
    A = [K, F];
    
    % --- Forward Elimination ---
    for i = 1:n-1
        % Partial Pivoting: Find the row with the largest pivot
        [~, max_idx] = max(abs(A(i:n, i)));
        max_row = max_idx + i - 1;
        
        % Swap rows if necessary to prevent division by zero / instability
        if max_row ~= i
            temp = A(i, :);
            A(i, :) = A(max_row, :);
            A(max_row, :) = temp;
        end
        
        % Check for singular matrix (if the pivot is numerically zero)
        if abs(A(i, i)) < 1e-12
            error('Matrix is structurally singular. Check boundary conditions.');
        end
        
        % Eliminate lower entries
        for j = i+1:n
            factor = A(j, i) / A(i, i);
            A(j, i:end) = A(j, i:end) - factor * A(i, i:end);
        end
    end
    
    % --- Back Substitution ---
    for i = n:-1:1
        U(i) = (A(i, end) - A(i, i+1:n) * U(i+1:n)) / A(i, i);
    end
end