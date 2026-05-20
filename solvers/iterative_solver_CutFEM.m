function [U, iter, res_history] = iterative_solver_CutFEM(mesh, conn, edof, dof, Re, dt, U_guess, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, tol, max_iter, phi, classification, ghost_info, cutfem_params)
% ITERATIVE_SOLVER_CUTFEM Non-linear Picard solver for CutFEM Navier-Stokes
% using BiCGStab with ILU preconditioning for the linear system solves.

    U_k = U_guess;
    res_history = [];
    fprintf('Starting Iterative Picard Solver (CutFEM)...\n');

    for iter = 1:max_iter
        % 1. Assemble monolithic LHS and RHS
        [I, J, V, F_raw] = navierstokes_cutfem(mesh, conn, edof, dof, Re, dt, U_k, U_n, f_source, phi, classification, ghost_info, cutfem_params);
        
        K = sparse(I, J, V, dof.ndof, dof.ndof);
        K = K / dt;
        F_raw = F_raw / dt;

        % 2. Apply Neumann Boundary Conditions
        if ~isempty(neumann_bcs)
            for b = 1:length(neumann_bcs)
                edges = neumann_bcs{b}{1};
                P_val = neumann_bcs{b}{2};
                tau_val = neumann_bcs{b}{3};
                if ~isempty(edges)
                    F_raw = neumannstokes(mesh, edges, dof, F_raw, P_val, tau_val);
                end
            end
        end

        % 3. Apply Dirichlet Boundary Conditions
        [K_mod, F_mod] = dirichlet(K, F_raw, bc_dofs, bc_vals);
        
        % 4. Solve the Linear System (BiCGStab + ILU(0))
        fprintf('  Solving linear system with BiCGStab... ');
        
        % Use a strict inner tolerance to ensure pressure (mass conservation) is accurately resolved
        inner_tol = 1e-7;
        inner_maxit = 1000;
        
        % --- AMD-Permuted ILUTP Preconditioner ---
        % For high-order Q9 elements, the matrix condition number is extremely high.
        % ILU(0) (nofill) is too weak to capture the dense stencil, causing BiCGStab to stall.
        % However, computing ILUTP on the raw matrix is too slow due to fill-in.
        % The optimal solution: Apply ILUTP with a drop tolerance ON the AMD-permuted matrix!
        p = symamd(K_mod); 
        
        setup.type = 'ilutp';
        setup.droptol = 1e-3; % 1e-3 strikes the perfect balance between fast computation and strong preconditioning
        
        try
            % 1. Compute ILUTP on the permuted matrix (Computes instantly compared to raw matrix)
            [L, U_pre] = ilu(K_mod(p, p), setup);
            
            % 2. Solve the permuted linear system
            [U_p, flag, relres, it] = bicgstab(K_mod(p, p), F_mod(p), inner_tol, inner_maxit, L, U_pre, U_k(p));
            
            % 3. Reverse the permutation to get the physical velocity/pressure field
            U_new = zeros(size(U_k));
            U_new(p) = U_p;
            
        catch ME
            fprintf('\n  [!] Fast ILUTP failed: %s\n  Falling back to un-preconditioned...\n', ME.message);
            [U_new, flag, relres, it] = bicgstab(K_mod, F_mod, inner_tol, inner_maxit, [], [], U_k);
        end
        
        if flag == 0
            fprintf('Converged in %d iters (Relres: %e)\n', it, relres);
        else
            fprintf('Failed to converge cleanly. Flag: %d, Relres: %e, Iters: %d\n', flag, relres, it);
        end

        % 5. Check Non-Linear Convergence
        dU     = norm(U_new - U_k);
        normU  = max(norm(U_new), 1e-12);
        rel_error = dU / normU;
        res_history = [res_history; rel_error];

        if rel_error < tol
            fprintf('Picard converged in %d iterations.\n', iter);
            U = U_new;
            return;
        end
        U_k = U_new;
    end
    fprintf('Warning: Picard failed to converge within %d iterations.\n', max_iter);
    U = U_k;
end
