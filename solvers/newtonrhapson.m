function [U, iter, res_history] = newtonrhapson(mesh, conn, edof, dof, Re, dt, U_guess, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, tol, max_iter)
% NEWTONRHAPSON Solves the non-linear Navier-Stokes equations using the
% analytical Jacobian.  GPU-accelerated assembly via navierstokes_jacobian.

    U_k = U_guess;
    res_history = [];
    fprintf('Starting Newton-Raphson Iterations...\n');

    for iter = 1:max_iter
        [I, J_idx, V, R_raw] = navierstokes_jacobian(mesh, conn, edof, dof, Re, dt, U_k, U_n, f_source);
        J_mat = sparse(I, J_idx, V, dof.ndof, dof.ndof);
        J_mat = J_mat / dt;
        R_raw = R_raw / dt;

        if ~isempty(neumann_bcs)
            for b = 1:length(neumann_bcs)
                edges = neumann_bcs{b}{1};
                P_val = neumann_bcs{b}{2};
                tau_val = neumann_bcs{b}{3};
                if ~isempty(edges)
                    F_neu = neumannstokes(mesh, edges, dof, zeros(dof.ndof,1), P_val, tau_val);
                    R_raw = R_raw - F_neu;
                end
            end
        end

        [J_mod, R_mod] = dirichlet_jacobian(J_mat, R_raw, bc_dofs, bc_vals, U_k);
        dU = J_mod \ -R_mod;
        U_new = U_k + dU;

        err = norm(dU) / max(norm(U_new), 1e-12);
        res_history = [res_history; err];
        fprintf('  Iter %3d: Rel Error = %e\n', iter, err);

        if err < tol
            fprintf('Newton-Raphson converged in %d iterations.\n', iter);
            U = U_new;
            return;
        end
        U_k = U_new;
    end
    fprintf('Warning: Newton-Raphson failed to converge within %d iterations.\n', max_iter);
    U = U_k;
end