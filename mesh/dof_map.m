function dof = dof_map(mesh, ndpn)

nnode = size(mesh.nodes,1);

dof.node = zeros(nnode, ndpn);

counter = 1;
for i=1:nnode
    for j=1:ndpn
        dof.node(i,j) = counter;
        counter = counter + 1;
    end
end

dof.ndof = counter-1;

end
