function plot_streamlines(mesh, dof, U)
    U = gather(U);
    x = mesh.nodes(:, 1);
    y = mesh.nodes(:, 2);
    u = U(dof.node(:, 1));
    v = U(dof.node(:, 2));
    res = 150;
    xi = linspace(min(x), max(x), res);
    yi = linspace(min(y), max(y), res);
    [X, Y] = meshgrid(xi, yi);
    U_grid = griddata(x, y, u, X, Y, 'natural');
    V_grid = griddata(x, y, v, X, Y, 'natural');
    h_streams = streamslice(X, Y, U_grid, V_grid, 2);
    set(h_streams, 'Color', 'w', 'LineWidth', 1.0);
end