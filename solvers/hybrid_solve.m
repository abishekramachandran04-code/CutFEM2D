function [U, total_iters, res_history] = hybrid_solve(mesh, conn, edof, dof, Re, dt, U_guess, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, tol_picard, max_iter_picard, tol_nr, max_iter_nr)

    fprintf('\n=== Starting Hybrid Non-Linear Solver ===\n');
    [U_picard, iters_p, res_p] = picard(mesh, conn, edof, dof, Re, dt, U_guess, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, tol_picard, max_iter_picard);

    if res_p(end) > tol_picard
        fprintf('  -> Picard hit max iterations. Handing to NR...\n');
    else
        fprintf('  -> Picard reached target. Final NR refinement...\n');
    end

    [U, iters_nr, res_nr] = newtonrhapson(mesh, conn, edof, dof, Re, dt, U_picard, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, tol_nr, max_iter_nr);

    total_iters = iters_p + iters_nr;
    res_history = [res_p; res_nr];
    fprintf('=== Hybrid Solver Completed in %d Total Iterations ===\n\n', total_iters);
end