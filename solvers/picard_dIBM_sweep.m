function [U_new, iter, error_hist] = picard_dIBM_sweep(mesh, conn, edof, dof, Re, dt, U_n, U_guess, f_source, bc_dofs, bc_vals, neumann_bcs, tol, max_iter, c_eps, c_k)
% PICARD_DIBM_SWEEP  Identical to picard.m but calls navierstokes_dIBM_sweep.

    U_k = U_guess;
    error_hist = []; % [FIX 1]: Renamed from res_history to error_hist
    
    % fprintf('Starting Picard Iterations (dIBM)...\n'); % Optional: mute this so it doesn't spam your terminal 64 times
    
    for iter = 1:max_iter
        [I, J, V, F_raw] = navierstokes_dIBM_sweep(mesh, conn, edof, dof, Re, dt, U_k, U_n, f_source, c_eps, c_k);
        K = sparse(I, J, V, dof.ndof, dof.ndof);
        K = K / dt;
        F_raw = F_raw / dt;
        
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
        
        [K_mod, F_mod] = dirichlet(K, F_raw, bc_dofs, bc_vals);
        U_new = K_mod \ F_mod;
        
        dU     = norm(U_new - U_k);
        normU  = max(norm(U_new), 1e-12);
        rel_error = dU / normU;
        
        error_hist = [error_hist; rel_error]; % [FIX 1]: Using error_hist
        
        if rel_error < tol
            % fprintf('Picard converged in %d iterations.\n', iter);
            return; % [FIX 2]: Removed "U = U_new", just return directly since U_new is already correct
        end
        U_k = U_new;
    end
    
    fprintf('Warning: Picard failed to converge within %d iterations.\n', max_iter);
    % [FIX 2]: Removed "U = U_k", U_new is the output variable.
end