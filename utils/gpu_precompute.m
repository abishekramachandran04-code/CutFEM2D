function cache = gpu_precompute(mesh, conn, edof, dof)
% GPU_PRECOMPUTE  Upload all static data to GPU ONCE (double precision).
% Precomputes shape functions, element geometry, and sparse index patterns.

    cache.mesh = mesh;
    cache.dof  = dof;
    cache.total_dof = dof.ndof;
    ndpn = 3;

    nodes_g = gpuArray(mesh.nodes(:,1:2));   % double precision
    elem_types = {'tri3','quad4','tri6','quad9'};
    I_all = [];  J_all = [];

    for t = 1:length(elem_types)
        type = elem_types{t};
        if ~isfield(conn,type) || isempty(conn.(type))
            cache.(type).active = false;
            continue;
        end
        cache.(type).active = true;

        elems = conn.(type);  edofs_mat = edof.(type);
        ne = size(elems,1);  nen = size(elems,2);  ndof_e = nen*ndpn;
        cache.(type).ne = ne;  cache.(type).nen = nen;  cache.(type).ndof_e = ndof_e;
        cache.(type).is_quad = (nen==6 || nen==9);

        elems_g = gpuArray(elems);
        edofs_g = gpuArray(edofs_mat);
        cache.(type).edofs_g = edofs_g;

        % Coordinates [nen x 2 x ne]
        idx_f  = reshape(elems_g',[], 1);
        crd_f  = nodes_g(idx_f, :);
        coords = permute(reshape(crd_f, nen, ne, 2), [1 3 2]);
        cache.(type).coords = coords;

        % DOF extraction index
        cache.(type).ed_flat = reshape(edofs_g',[], 1);

        % Force scatter index (CPU)
        cache.(type).ed_cpu = gather(edofs_g');   % [ndof_e x ne]

        % Gauss quadrature
        [xi, eta, wt] = gauss_quadrature(type);
        nGP = length(wt);
        cache.(type).nGP = nGP;
        cache.(type).wt  = wt;

        % Shape functions at all GPs (double precision on GPU)
        N_all    = zeros(nen, nGP);
        dNxi_all = zeros(2, nen, nGP);
        for q = 1:nGP
            [N, dNxi] = shape_funcs(xi(q), eta(q), type);
            N_all(:,q) = N;  dNxi_all(:,:,q) = dNxi;
        end
        cache.(type).N_all    = gpuArray(N_all);
        cache.(type).dNxi_all = gpuArray(dNxi_all);

        % 2nd derivatives (quadratic only)
        if cache.(type).is_quad
            d2_all = zeros(3, nen, nGP);
            for q = 1:nGP, d2_all(:,:,q) = shape_funcs_2nd(xi(q),eta(q),type); end
            cache.(type).d2N_all = gpuArray(d2_all);
        end

        % Element areas & h_e (geometric constants — compute once)
        Ae = zeros(1,1,ne,'gpuArray');
        for q = 1:nGP
            Jq  = pagemtimes(cache.(type).dNxi_all(:,:,q), coords);
            dJq = Jq(1,1,:).*Jq(2,2,:) - Jq(1,2,:).*Jq(2,1,:);
            Ae  = Ae + dJq * wt(q);
        end
        if contains(type,'tri'), nc=3; else, nc=4; end
        corn = coords(1:nc,:,:);
        nxt  = [corn(2:end,:,:); corn(1,:,:)];
        cache.(type).h_e = 4*Ae ./ sum(sqrt(sum((nxt-corn).^2,2)),1);

        % Triplet index arrays (constant pattern — precompute on CPU)
        r0 = repmat((1:ndof_e)',1,ndof_e);
        c0 = repmat(1:ndof_e, ndof_e,1);
        r0 = r0(:);  c0 = c0(:);
        edofs_cpu = gather(edofs_g);
        I_t = edofs_cpu(:,r0)';   J_t = edofs_cpu(:,c0)';

        cache.(type).V_offset = length(I_all);
        cache.(type).V_count  = ndof_e^2 * ne;

        I_all = [I_all; I_t(:)];
        J_all = [J_all; J_t(:)];
    end

    % Sparse pattern condensation (precompute unique mapping once)
    [ij_u, ~, ic] = unique([I_all, J_all], 'rows');
    cache.I_u = ij_u(:,1);
    cache.J_u = ij_u(:,2);
    cache.ic  = ic;
    cache.n_triplets = length(I_all);

    fprintf('GPU cache ready: %d unique sparse entries from %d triplets\n', ...
        length(cache.I_u), cache.n_triplets);
end
