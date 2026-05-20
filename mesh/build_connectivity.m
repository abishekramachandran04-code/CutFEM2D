function conn = build_connectivity(mesh)

conn.tri3  = mesh.tri3;
conn.quad4 = mesh.quad4;
conn.tri6  = mesh.tri6;
conn.quad9 = mesh.quad9;

conn.nnode = size(mesh.nodes,1);

end
