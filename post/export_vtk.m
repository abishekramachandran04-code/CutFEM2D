function export_vtk(filename, mesh, conn, dof, U, phi)
% EXPORT_VTK Exports mesh and results to a legacy VTK format for ParaView.
% 
% Inputs:
%   filename: Output file name (e.g., 'results_001.vtk')
%   mesh:     Mesh struct with .nodes
%   conn:     Connectivity struct
%   dof:      DOF map
%   U:        Solution vector [u, v, p]
%   phi:      Level-set function at nodes

    num_nodes = size(mesh.nodes, 1);
    
    % Extract physical variables
    u = U(dof.node(:, 1));
    v = U(dof.node(:, 2));
    p = U(dof.node(:, 3));

    % Open file
    fid = fopen(filename, 'w');
    if fid == -1
        error('Cannot open file %s for writing.', filename);
    end

    % 1. Write Header
    fprintf(fid, '# vtk DataFile Version 3.0\n');
    fprintf(fid, 'CutFEM Results\n');
    fprintf(fid, 'ASCII\n');
    fprintf(fid, 'DATASET UNSTRUCTURED_GRID\n');

    % 2. Write Points
    fprintf(fid, 'POINTS %d float\n', num_nodes);
    % VTK requires 3D coordinates. We append z = 0.
    points = [mesh.nodes(:, 1:2), zeros(num_nodes, 1)];
    fprintf(fid, '%f %f %f\n', points');

    % 3. Collect Cells
    cell_data = [];
    cell_types = [];
    num_cells = 0;
    
    % Element mapping to VTK types
    % VTK_TRIANGLE = 5, VTK_QUAD = 9, VTK_QUADRATIC_TRIANGLE = 22, VTK_BIQUADRATIC_QUAD = 28
    
    if isfield(conn, 'tri3') && ~isempty(conn.tri3)
        ne = size(conn.tri3, 1);
        num_cells = num_cells + ne;
        cell_types = [cell_types; repmat(5, ne, 1)];
        % Format: [num_nodes, n1, n2, n3] (VTK uses 0-based indexing)
        c = [repmat(3, ne, 1), conn.tri3 - 1];
        cell_data = [cell_data; c'];
    end
    
    if isfield(conn, 'quad4') && ~isempty(conn.quad4)
        ne = size(conn.quad4, 1);
        num_cells = num_cells + ne;
        cell_types = [cell_types; repmat(9, ne, 1)];
        c = [repmat(4, ne, 1), conn.quad4 - 1];
        cell_data = [cell_data; c'];
    end
    
    if isfield(conn, 'tri6') && ~isempty(conn.tri6)
        ne = size(conn.tri6, 1);
        num_cells = num_cells + ne;
        cell_types = [cell_types; repmat(22, ne, 1)];
        c = [repmat(6, ne, 1), conn.tri6 - 1];
        cell_data = [cell_data; c'];
    end
    
    if isfield(conn, 'quad9') && ~isempty(conn.quad9)
        ne = size(conn.quad9, 1);
        num_cells = num_cells + ne;
        cell_types = [cell_types; repmat(28, ne, 1)];
        c = [repmat(9, ne, 1), conn.quad9 - 1];
        cell_data = [cell_data; c'];
    end

    % 4. Write Cells
    % Total size of cell_data array: num_cells + total number of nodes in cells
    total_cell_size = numel(cell_data);
    fprintf(fid, '\nCELLS %d %d\n', num_cells, total_cell_size);
    
    % Print cell data based on number of nodes. 
    % We process cell_data sequentially based on the structure we built.
    idx = 1;
    for i = 1:num_cells
        n_pts = cell_data(idx);
        % Format string dynamically: e.g., '%d %d %d %d\n'
        fmt = [repmat('%d ', 1, n_pts + 1), '\n'];
        fprintf(fid, fmt, cell_data(idx : idx + n_pts));
        idx = idx + n_pts + 1;
    end

    % 5. Write Cell Types
    fprintf(fid, '\nCELL_TYPES %d\n', num_cells);
    fprintf(fid, '%d\n', cell_types);

    % 6. Write Point Data
    fprintf(fid, '\nPOINT_DATA %d\n', num_nodes);
    
    % Velocity (Vector)
    fprintf(fid, 'VECTORS velocity float\n');
    vel = [u, v, zeros(num_nodes, 1)];
    fprintf(fid, '%f %f %f\n', vel');
    
    % Pressure (Scalar)
    fprintf(fid, '\nSCALARS pressure float 1\n');
    fprintf(fid, 'LOOKUP_TABLE default\n');
    fprintf(fid, '%f\n', p);
    
    % Level Set (Scalar) - Extremely useful for visualizing the immersed boundary!
    fprintf(fid, '\nSCALARS levelset float 1\n');
    fprintf(fid, 'LOOKUP_TABLE default\n');
    fprintf(fid, '%f\n', phi);

    fclose(fid);
end
