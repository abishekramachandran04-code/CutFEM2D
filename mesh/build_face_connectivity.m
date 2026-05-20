function faces = build_face_connectivity(conn)
% BUILD_FACE_CONNECTIVITY  Builds internal face (edge) connectivity.
%   faces = build_face_connectivity(conn)
%
%   Output:
%       faces.edges     — [nFaces x 2] global node pairs defining each face
%       faces.elem_L    — [nFaces x 1] left element global index
%       faces.elem_R    — [nFaces x 1] right element global index (0 = boundary)
%       faces.type_L    — [nFaces x 1] cell array of element type strings
%       faces.type_R    — [nFaces x 1] cell array of element type strings
%       faces.local_L   — [nFaces x 1] local element index within its type
%       faces.local_R   — [nFaces x 1] local element index within its type

    types = {'tri3', 'quad4', 'tri6', 'quad9'};
    corner_nodes = [3, 4, 3, 4];  % Number of corner nodes per type

    % Edge definitions (local corner node pairs) for each type
    edge_defs = {
        [1 2; 2 3; 3 1],           % tri3
        [1 2; 2 3; 3 4; 4 1],      % quad4
        [1 2; 2 3; 3 1],           % tri6 (corner edges only)
        [1 2; 2 3; 3 4; 4 1]       % quad9 (corner edges only)
    };

    % Build a hash map: sorted edge pair -> list of (type, local_elem_idx)
    edge_map = containers.Map('KeyType', 'char', 'ValueType', 'any');

    for t = 1:length(types)
        name = types{t};
        if ~isfield(conn, name) || isempty(conn.(name)), continue; end

        elems = conn.(name);
        ne = size(elems, 1);
        edefs = edge_defs{t};

        for e = 1:ne
            nodes_e = elems(e, :);
            for ed = 1:size(edefs, 1)
                n1 = nodes_e(edefs(ed, 1));
                n2 = nodes_e(edefs(ed, 2));
                key = sprintf('%d-%d', min(n1,n2), max(n1,n2));

                entry.type = name;
                entry.local_idx = e;
                entry.nodes = [n1, n2];

                if isKey(edge_map, key)
                    edge_map(key) = [edge_map(key), entry];
                else
                    edge_map(key) = entry;
                end
            end
        end
    end

    % Extract internal faces (shared by exactly 2 elements)
    keys = edge_map.keys();
    nKeys = length(keys);

    edge_list = zeros(nKeys, 2);
    elem_L    = zeros(nKeys, 1);
    elem_R    = zeros(nKeys, 1);
    type_L    = cell(nKeys, 1);
    type_R    = cell(nKeys, 1);
    local_L   = zeros(nKeys, 1);
    local_R   = zeros(nKeys, 1);
    count = 0;

    for i = 1:nKeys
        entries = edge_map(keys{i});
        if length(entries) == 2
            count = count + 1;
            edge_list(count, :) = sort(entries(1).nodes);
            type_L{count}  = entries(1).type;
            local_L(count) = entries(1).local_idx;
            type_R{count}  = entries(2).type;
            local_R(count) = entries(2).local_idx;
        end
    end

    % Trim to actual count
    faces.edges   = edge_list(1:count, :);
    faces.type_L  = type_L(1:count);
    faces.type_R  = type_R(1:count);
    faces.local_L = local_L(1:count);
    faces.local_R = local_R(1:count);
end
