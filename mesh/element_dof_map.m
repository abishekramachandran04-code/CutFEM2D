function edof = element_dof_map(conn, dof)

ndpn = size(dof.node,2);

types = {'tri3','quad4','tri6','quad9'};

for t = 1:length(types)

    name = types{t};

    if isempty(conn.(name))
        edof.(name) = [];
        continue
    end

    elems = conn.(name);
    ne = size(elems,1);
    nen = size(elems,2);

    ed = zeros(ne, nen*ndpn);

    for e=1:ne
        list = [];
        for a=1:nen
            n = elems(e,a);
            list = [list dof.node(n,:)];
        end
        ed(e,:) = list;
    end

    edof.(name) = ed;

end

end
