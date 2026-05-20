% NS_CutFEM_2.m
% Runs Standard NS (body-fitted) and CutFEM side by side,
% computing per-timestep L2 relative errors over ALL standard mesh nodes.
% Also computes and prints Cd/Cl for both solvers at each timestep.
clear; clc; close all;
addpath('mesh','fem','physics','utils','bc','solvers','post');

gpu_dev = gpu_init();

%% ============ MESH SETUP ============
fprintf('=============== MESH SETUP ===============\n');

% 1. Standard body-fitted mesh (the "truth")
meshfile_std = fullfile('gmsh','NSmesh_Q4.msh');
fprintf('Loading Standard Mesh: %s\n', meshfile_std);
mesh_std = fix_orientation(read_gmsh(meshfile_std));
conn_std = build_connectivity(mesh_std);
dof_std  = dof_map(mesh_std, 3);
edof_std = element_dof_map(conn_std, dof_std);

% 2. CutFEM background mesh
meshfile_cut = fullfile('gmsh','channel2.msh');
fprintf('Loading CutFEM Mesh: %s\n', meshfile_cut);
mesh_cut = fix_orientation(read_gmsh(meshfile_cut));
conn_cut = build_connectivity(mesh_cut);

% CutFEM geometry
cx = 0.2; cy = 0.2; R_cyl = 0.05;
phi = eval_levelset(mesh_cut.nodes, cx, cy, R_cyl);
classification = classify_elements(conn_cut, phi);
faces = build_face_connectivity(conn_cut);
ghost_info = build_ghost_faces(faces, classification);

dof_cut  = dof_map(mesh_cut, 3);
edof_cut = element_dof_map(conn_cut, dof_cut);

%% ============ PHYSICS ============
fprintf('Setting up Physics...\n');
Re_D = 100; H = 0.41; U_max = 1.5; U_mean = (2/3)*U_max;
D = 2*R_cyl; nu = U_mean*D/Re_D; Re = 1/nu;
f_source = {@(x,y) 0, @(x,y) 0, @(x,y) 0};

dt = 0.005; t_end = 30; 
time_steps = ceil(t_end / dt);

% CutFEM parameters
cutfem_params.alpha_v = 0.05;
cutfem_params.alpha_p = 0.1;
cutfem_params.alpha_adv = 0.01;
cutfem_params.gamma_u = 40;  % Effective penalty = gamma_u/(Re*h) ≈ 40

%% ============ BCs: STANDARD ============
bc_dofs_std = []; bc_vals_std = [];
inlet_nodes_std = get_boundary_nodes(mesh_std, {'inlet'});
for i = 1:length(inlet_nodes_std)
    y_n = mesh_std.nodes(inlet_nodes_std(i), 2);
    bc_dofs_std = [bc_dofs_std; dof_std.node(inlet_nodes_std(i),1)]; bc_vals_std = [bc_vals_std; 4*U_max*y_n*(H-y_n)/H^2];
    bc_dofs_std = [bc_dofs_std; dof_std.node(inlet_nodes_std(i),2)]; bc_vals_std = [bc_vals_std; 0];
end
wall_nodes_std = get_boundary_nodes(mesh_std, {'walltop','wallbottom'});
for i = 1:length(wall_nodes_std)
    bc_dofs_std = [bc_dofs_std; dof_std.node(wall_nodes_std(i),1); dof_std.node(wall_nodes_std(i),2)];
    bc_vals_std = [bc_vals_std; 0; 0];
end
cyl_nodes_std = get_boundary_nodes(mesh_std, {'wallcylinder'});
for i = 1:length(cyl_nodes_std)
    bc_dofs_std = [bc_dofs_std; dof_std.node(cyl_nodes_std(i),1); dof_std.node(cyl_nodes_std(i),2)];
    bc_vals_std = [bc_vals_std; 0; 0];
end
neumann_bcs_std = { {mesh_std.boundaries.outlet.edges, 0, 0} };

%% ============ BCs: CUTFEM ============
bc_dofs_cut_base = []; bc_vals_cut_base = [];
inlet_nodes_cut = get_boundary_nodes(mesh_cut, {'inlet'});
inlet_u_dofs = []; inlet_u_base = [];
for i = 1:length(inlet_nodes_cut)
    y_n = mesh_cut.nodes(inlet_nodes_cut(i), 2);
    inlet_u_dofs = [inlet_u_dofs; dof_cut.node(inlet_nodes_cut(i),1)];
    inlet_u_base = [inlet_u_base; 4*U_max*y_n*(H-y_n)/H^2];
    bc_dofs_cut_base = [bc_dofs_cut_base; dof_cut.node(inlet_nodes_cut(i),2)];
    bc_vals_cut_base = [bc_vals_cut_base; 0];
end
wall_nodes_cut = get_boundary_nodes(mesh_cut, {'walltop','wallbottom'});
for i = 1:length(wall_nodes_cut)
    bc_dofs_cut_base = [bc_dofs_cut_base; dof_cut.node(wall_nodes_cut(i),1); dof_cut.node(wall_nodes_cut(i),2)];
    bc_vals_cut_base = [bc_vals_cut_base; 0; 0];
end
% Solid node dummy BCs
active_nodes = [];
elem_types = {'tri3','quad4','tri6','quad9'};
for t = 1:4
    type = elem_types{t};
    if isfield(classification, type)
        ae = conn_cut.(type)([classification.(type).FULL_FLUID; classification.(type).CUT], :);
        active_nodes = [active_nodes; ae(:)];
    end
end
solid_nodes = setdiff((1:size(mesh_cut.nodes,1))', unique(active_nodes));
for i = 1:length(solid_nodes)
    bc_dofs_cut_base = [bc_dofs_cut_base; dof_cut.node(solid_nodes(i),1); dof_cut.node(solid_nodes(i),2); dof_cut.node(solid_nodes(i),3)];
    bc_vals_cut_base = [bc_vals_cut_base; 0; 0; 0];
end
neumann_bcs_cut = { {mesh_cut.boundaries.outlet.edges, 0, 0} };

%% ============ INITIALIZATION ============
U_n_std = set_ic(dof_std, U_mean, 0, 0);
U_n_cut = set_ic(dof_cut, U_mean, 0, 0);
% Zero out velocity inside the cylinder for CutFEM
for i = 1:size(mesh_cut.nodes, 1)
    if sqrt((mesh_cut.nodes(i,1)-cx)^2 + (mesh_cut.nodes(i,2)-cy)^2) <= R_cyl
        U_n_cut(dof_cut.node(i,1)) = 0;
        U_n_cut(dof_cut.node(i,2)) = 0;
    end
end

tol_picard = 1e-6; max_iter_picard = 100;

% Error storage
err_u_hist = zeros(time_steps, 1);
err_v_hist = zeros(time_steps, 1);
err_p_hist = zeros(time_steps, 1);
t_hist = zeros(time_steps, 1);

% Cd/Cl storage
CD_std_hist = zeros(time_steps, 1);
CL_std_hist = zeros(time_steps, 1);
CD_cut_hist = zeros(time_steps, 1);
CL_cut_hist = zeros(time_steps, 1);

%% ============ TIME STEPPING ============
fprintf('\n=============== STARTING TIME STEPPING ===============\n');
for t_step = 1:time_steps
    current_time = t_step * dt;
    t_hist(t_step) = current_time;
    
    % --- Standard NS ---
    [U_new_std, it_std, ~] = picard(mesh_std, conn_std, edof_std, dof_std, Re, dt, ...
        U_n_std, U_n_std, f_source, bc_dofs_std, bc_vals_std, neumann_bcs_std, tol_picard, max_iter_picard);
    
    % --- CutFEM NS (same inlet as standard, no ramp) ---
    bc_dofs_cut = [inlet_u_dofs; bc_dofs_cut_base];
    bc_vals_cut = [inlet_u_base; bc_vals_cut_base];
    
    [U_new_cut, it_cut, ~] = picard_cutfem(mesh_cut, conn_cut, edof_cut, dof_cut, Re, dt, ...
        U_n_cut, U_n_cut, f_source, bc_dofs_cut, bc_vals_cut, neumann_bcs_cut, ...
        tol_picard, max_iter_picard, phi, classification, ghost_info, cutfem_params);
    
    % --- Aerodynamic Forces: Standard (reaction force method) ---
    [~, ~, CD_s, CL_s] = calc_aero_forces(mesh_std, conn_std, edof_std, dof_std, Re, dt, U_new_std, U_n_std, f_source, cyl_nodes_std, D, U_mean);
    CD_std_hist(t_step) = CD_s;
    CL_std_hist(t_step) = CL_s;
    
    % --- Aerodynamic Forces: CutFEM (Nitsche boundary integral) ---
    [~, ~, CD_c, CL_c] = calc_aero_forces_CutFEM(mesh_cut, conn_cut, edof_cut, dof_cut, Re, U_new_cut, ...
        phi, classification, cutfem_params, D, U_mean);
    CD_cut_hist(t_step) = CD_c;
    CL_cut_hist(t_step) = CL_c;
    
    % --- Interpolate CutFEM solution onto Standard mesh nodes ---
    u_std = full(gather(U_new_std(1:3:end)));
    v_std = full(gather(U_new_std(2:3:end)));
    p_std = full(gather(U_new_std(3:3:end)));
    
    u_cut = U_new_cut(1:3:end);
    v_cut = U_new_cut(2:3:end);
    p_cut = U_new_cut(3:3:end);
    
    % Only interpolate from FLUID-side CutFEM nodes (phi < 0) to avoid
    % ghost-penalty contaminated solid-side values polluting the interpolant
    fluid_mask = phi < 0;
    F_u = scatteredInterpolant(mesh_cut.nodes(fluid_mask,1), mesh_cut.nodes(fluid_mask,2), u_cut(fluid_mask), 'natural', 'none');
    F_v = scatteredInterpolant(mesh_cut.nodes(fluid_mask,1), mesh_cut.nodes(fluid_mask,2), v_cut(fluid_mask), 'natural', 'none');
    F_p = scatteredInterpolant(mesh_cut.nodes(fluid_mask,1), mesh_cut.nodes(fluid_mask,2), p_cut(fluid_mask), 'natural', 'none');
    
    u_cut_interp = F_u(mesh_std.nodes(:,1), mesh_std.nodes(:,2));
    v_cut_interp = F_v(mesh_std.nodes(:,1), mesh_std.nodes(:,2));
    p_cut_interp = F_p(mesh_std.nodes(:,1), mesh_std.nodes(:,2));
    
    % L2 relative error over all valid standard mesh nodes
    valid = ~isnan(u_cut_interp);
    err_u_hist(t_step) = norm(u_std(valid) - u_cut_interp(valid)) / (norm(u_std(valid)) + 1e-12);
    err_v_hist(t_step) = norm(v_std(valid) - v_cut_interp(valid)) / (norm(v_std(valid)) + 1e-12);
    err_p_hist(t_step) = norm(p_std(valid) - p_cut_interp(valid)) / (norm(p_std(valid)) + 1e-12);
    
    % L2 relative error over near-cylinder region (Cut cells + 2-3 neighbors)
    % Define region as r <= R_cyl + 0.05 (approx 5 cell widths depending on mesh)
    r_std = sqrt((mesh_std.nodes(:,1)-cx).^2 + (mesh_std.nodes(:,2)-cy).^2);
    near_mask = valid & (r_std <= R_cyl + 0.01);
    
    err_u_near = norm(u_std(near_mask) - u_cut_interp(near_mask)) / (norm(u_std(near_mask)) + 1e-12);
    err_v_near = norm(v_std(near_mask) - v_cut_interp(near_mask)) / (norm(v_std(near_mask)) + 1e-12);
    err_p_near = norm(p_std(near_mask) - p_cut_interp(near_mask)) / (norm(p_std(near_mask)) + 1e-12);
    
    fprintf('Step %2d/%d (t=%.3f) | Std:%2d it | Cut:%2d it\n', t_step, time_steps, current_time, it_std, it_cut);
    fprintf('  Global L2err -> u:%6.2f%% | v:%6.2f%% | p:%6.2f%%\n', 100*err_u_hist(t_step), 100*err_v_hist(t_step), 100*err_p_hist(t_step));
    fprintf('  Near-Bnd err -> u:%6.2f%% | v:%6.2f%% | p:%6.2f%%\n', 100*err_u_near, 100*err_v_near, 100*err_p_near);
    fprintf('  Std -> C_D = %.4f | C_L = %.4f   ||   Cut -> C_D = %.4f | C_L = %.4f\n', CD_s, CL_s, CD_c, CL_c);
    
    U_n_std = U_new_std;
    U_n_cut = U_new_cut;
end

%% ============ SUMMARY ============
fprintf('\n=============== FINAL RESULTS ===============\n');
fprintf('FINAL L2 ERROR -> u: %5.2f%% | v: %5.2f%% | p: %5.2f%%\n', ...
    100*err_u_hist(end), 100*err_v_hist(end), 100*err_p_hist(end));
fprintf('FINAL Std  -> C_D = %.4f | C_L = %.4f\n', CD_std_hist(end), CL_std_hist(end));
fprintf('FINAL Cut  -> C_D = %.4f | C_L = %.4f\n', CD_cut_hist(end), CL_cut_hist(end));

%% ============ ERROR CONVERGENCE PLOT ============
figure('Name','CutFEM vs Standard: Error History','Position',[100,100,700,450]);
semilogy(t_hist, 100*err_u_hist, 'b-o','LineWidth',1.5,'MarkerSize',4); hold on;
semilogy(t_hist, 100*err_v_hist, 'r-s','LineWidth',1.5,'MarkerSize',4);
semilogy(t_hist, 100*err_p_hist, '-^','Color',[0.2 0.9 0.2],'LineWidth',1.5,'MarkerSize',4);
xlabel('Time (s)'); ylabel('L_2 Relative Error (%)');
legend('u-velocity','v-velocity','pressure','Location','best');
title('CutFEM vs Body-Fitted: Per-Timestep L_2 Error');
grid on; hold off;

%% ============ Cd/Cl COMPARISON PLOTS ============
figure('Name','Cd/Cl: Standard vs CutFEM','Position',[100,550,800,500]);

subplot(2,1,1);
plot(t_hist, CD_std_hist, 'b-', 'LineWidth', 1.5); hold on;
plot(t_hist, CD_cut_hist, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('C_D');
legend('Standard (body-fitted)','CutFEM','Location','best');
title('Drag Coefficient Comparison'); grid on; hold off;

subplot(2,1,2);
plot(t_hist, CL_std_hist, 'b-', 'LineWidth', 1.5); hold on;
plot(t_hist, CL_cut_hist, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('C_L');
legend('Standard (body-fitted)','CutFEM','Location','best');
title('Lift Coefficient Comparison'); grid on; hold off;

%% ============ FINAL FIELD PLOTS ============
theta_plot = linspace(0,2*pi,100);

figure('Name','Standard u','Position',[50, 500, 500, 400]);
plot_results(mesh_std, conn_std, dof_std, U_new_std, 1, []);
title(sprintf('Standard u-Velocity | t=%.3f', t_end));

figure('Name','CutFEM u','Position',[570, 500, 500, 400]);
plot_results(mesh_cut, conn_cut, dof_cut, U_new_cut, 1, []);
hold on; plot(cx+R_cyl*cos(theta_plot), cy+R_cyl*sin(theta_plot),'k-','LineWidth',2); hold off;
title(sprintf('CutFEM u-Velocity | t=%.3f', t_end));

figure('Name','Standard p','Position',[50, 50, 500, 400]);
plot_results(mesh_std, conn_std, dof_std, U_new_std, 3, []);
title(sprintf('Standard Pressure | t=%.3f', t_end));

figure('Name','CutFEM p','Position',[570, 50, 500, 400]);
plot_results(mesh_cut, conn_cut, dof_cut, U_new_cut, 3, []);
hold on; plot(cx+R_cyl*cos(theta_plot), cy+R_cyl*sin(theta_plot),'k-','LineWidth',2); hold off;
title(sprintf('CutFEM Pressure | t=%.3f', t_end));

%% ============ 2D SPATIAL ERROR DISTRIBUTION ============
% Pointwise absolute error |std - cutfem_interp| over the standard mesh
err_u_nodal = abs(u_std - u_cut_interp);
err_v_nodal = abs(v_std - v_cut_interp);
err_p_nodal = abs(p_std - p_cut_interp);
err_u_nodal(~valid) = NaN;
err_v_nodal(~valid) = NaN;
err_p_nodal(~valid) = NaN;

% Get standard mesh element connectivity for patch plotting
std_elems = [];
types_check = {'quad4','tri3','quad9','tri6'};
for tt = 1:length(types_check)
    if isfield(conn_std, types_check{tt}) && ~isempty(conn_std.(types_check{tt}))
        std_elems = conn_std.(types_check{tt});
        break;
    end
end

figure('Name','Spatial Error: u','Position',[50, 500, 600, 350]);
patch('Faces', std_elems, 'Vertices', mesh_std.nodes(:,1:2), ...
    'FaceVertexCData', err_u_nodal, 'FaceColor','interp','EdgeColor','none');
hold on; plot(cx+R_cyl*cos(theta_plot), cy+R_cyl*sin(theta_plot),'k-','LineWidth',2); hold off;
colorbar; colormap('jet'); axis equal tight;
title(sprintf('|u_{std} - u_{cut}| at t=%.3f', t_end));
xlabel('X'); ylabel('Y');

figure('Name','Spatial Error: v','Position',[670, 500, 600, 350]);
patch('Faces', std_elems, 'Vertices', mesh_std.nodes(:,1:2), ...
    'FaceVertexCData', err_v_nodal, 'FaceColor','interp','EdgeColor','none');
hold on; plot(cx+R_cyl*cos(theta_plot), cy+R_cyl*sin(theta_plot),'k-','LineWidth',2); hold off;
colorbar; colormap('jet'); axis equal tight;
title(sprintf('|v_{std} - v_{cut}| at t=%.3f', t_end));
xlabel('X'); ylabel('Y');

figure('Name','Spatial Error: p','Position',[360, 50, 600, 350]);
patch('Faces', std_elems, 'Vertices', mesh_std.nodes(:,1:2), ...
    'FaceVertexCData', err_p_nodal, 'FaceColor','interp','EdgeColor','none');
hold on; plot(cx+R_cyl*cos(theta_plot), cy+R_cyl*sin(theta_plot),'k-','LineWidth',2); hold off;
colorbar; colormap('jet'); axis equal tight;
title(sprintf('|p_{std} - p_{cut}| at t=%.3f', t_end));
xlabel('X'); ylabel('Y');
