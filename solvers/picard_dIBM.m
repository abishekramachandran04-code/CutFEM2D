function [U, iter, res_history] = picard_dIBM(mesh, conn, edof, dof, Re, dt, U_guess, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, tol, max_iter)
% PICARD_DIBM  Identical to picard.m but calls navierstokes_dIBM.

    U_k = U_guess;
    res_history = [];
    fprintf('Starting Picard Iterations (dIBM)...\n');

    for iter = 1:max_iter
        [I, J, V, F_raw] = navierstokes_dIBM(mesh, conn, edof, dof, Re, dt, U_k, U_n, f_source);
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
        res_history = [res_history; rel_error];
        % fprintf('  Iter %3d: Rel Error = %e\n', iter, rel_error);

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
