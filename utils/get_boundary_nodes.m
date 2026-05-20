function nodes = get_boundary_nodes(mesh, boundary_names)
    % GET_BOUNDARY_NODES Aggregates unique node IDs for given boundary names.
    
    nodes = [];
    for b = 1:length(boundary_names)
        wall_name = boundary_names{b};
        if isfield(mesh.boundaries, wall_name)
            nodes = [nodes, mesh.boundaries.(wall_name).nodes];
        else
            warning('Boundary "%s" not found in the mesh structure.', wall_name);
        end
    end
    
    % Return unique nodes as a column vector to remove corner overlaps
    nodes = unique(nodes)'; 
end