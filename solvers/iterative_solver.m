function U = iterative_solver(K, F, method, tol, maxit)
    % ITERATIVE_SOLVER Solves sparse linear systems using Krylov subspace methods.
    %
    % Inputs:
    %   K      : Global sparse stiffness matrix
    %   F      : Global force vector
    %   method : 'pcg' (Symmetric) or 'bicgstab' (Non-Symmetric)
    %   tol    : Relative residual tolerance (default: 1e-6)
    %   maxit  : Maximum number of iterations (default: 5000)
    
    if nargin < 3, method = 'pcg'; end
    if nargin < 4, tol = 1e-6; end
    if nargin < 5, maxit = 5000; end

    total_dof = size(K, 1);
    U = zeros(total_dof, 1);
    
    % Preconditioner setup to accelerate convergence
    setup.type = 'nofill';
    
    switch lower(method)
        case 'pcg'
            % Preconditioned Conjugate Gradient (For Poisson / Symmetric)
            try
                L = ichol(K, setup); % Incomplete Cholesky Factorization
                [U, flag, relres, iter] = pcg(K, F, tol, maxit, L, L');
            catch
                % Fallback if ICHOL fails (e.g., matrix loses positive definiteness)
                disp('ICHOL preconditioning failed, running un-preconditioned PCG...');
                [U, flag, relres, iter] = pcg(K, F, tol, maxit);
            end
            
        case 'bicgstab'
            % Biconjugate Gradient Stabilized (For Navier-Stokes / Non-Symmetric)
            try
                [L, U_pre] = ilu(K, setup); % Incomplete LU Factorization
                [U, flag, relres, iter] = bicgstab(K, F, tol, maxit, L, U_pre);
            catch
                % Fallback if ILU fails
                disp('ILU preconditioning failed, running un-preconditioned BiCGSTAB...');
                [U, flag, relres, iter] = bicgstab(K, F, tol, maxit);
            end
            
        otherwise
            error('Unsupported iterative method: %s. Choose ''pcg'' or ''bicgstab''.', method);
    end
    
    % Solver Diagnostics
    if flag == 0
        fprintf('  -> %s converged in %d iterations (Rel. Residual: %e)\n', upper(method), iter, relres);
    else
        warning('%s did NOT converge. Flag: %d, Rel. Residual: %e, Iter: %d', upper(method), flag, relres, iter);
    end
end


%BiCGStab with jacobi preconditioning block jacobi ILU