function plot_results(mesh, conn, dof, U, target_dof, exact_sol_func)
    % PLOT_RESULTS Plots the FEM solution. Handles gpuArray inputs via gather().

    % Ensure CPU data for plotting
    U = gather(U);

    % 1. Extract the nodal values for the requested DOF
    U_nodal = U(dof.node(:, target_dof));

    % 2. Build a universal face matrix for the 'patch' plotter
    faces = [];
    if isfield(conn, 'tri3') && ~isempty(conn.tri3)
        faces = [faces; conn.tri3, NaN(size(conn.tri3, 1), 1)];
    end
    if isfield(conn, 'quad4') && ~isempty(conn.quad4)
        faces = [faces; conn.quad4];
    end
    if isfield(conn, 'tri6') && ~isempty(conn.tri6)
        faces = [faces; conn.tri6(:, 1:3), NaN(size(conn.tri6, 1), 1)];
    end
    if isfield(conn, 'quad9') && ~isempty(conn.quad9)
        faces = [faces; conn.quad9(:, 1:4)];
    end

    % 3. Plot the Numerical Solution Field
    patch('Vertices', mesh.nodes(:, 1:2), ...
          'Faces', faces, ...
          'FaceVertexCData', U_nodal, ...
          'FaceColor', 'interp', ...
          'EdgeColor', 'none');

    xlabel('X'); ylabel('Y');
    axis equal tight;
    colorbar;
    colormap('jet');

    % 4. Plot Error Fields (if exact function is provided)
    if nargin > 5 && ~isempty(exact_sol_func)
        x_coords = mesh.nodes(:, 1);
        y_coords = mesh.nodes(:, 2);

        U_exact_nodal = arrayfun(exact_sol_func, x_coords, y_coords);
        error_abs = abs(U_nodal - U_exact_nodal);
        error_sq  = (U_nodal - U_exact_nodal).^2;

        patch('Vertices', mesh.nodes(:, 1:2), 'Faces', faces, 'FaceVertexCData', error_abs, ...
              'FaceColor', 'interp', 'EdgeColor', 'none');
        axis equal tight; colorbar; colormap(flipud(hot));

        patch('Vertices', mesh.nodes(:, 1:2), 'Faces', faces, 'FaceVertexCData', error_sq, ...
              'FaceColor', 'interp', 'EdgeColor', 'none');
        axis equal tight; colorbar; colormap(flipud(hot));
    end
end