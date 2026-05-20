% mainCutFEM.m (Transient Navier-Stokes CutFEM: Flow Past an Immersed Cylinder)
clear; clc; close all;
addpath('mesh', 'fem', 'physics', 'utils', 'bc', 'solvers', 'post');

%% 0. GPU Initialization
gpu_dev = gpu_init();

%% 1. Mesh Processing and CutFEM Geometry Setup
disp('Reading Mesh and Building Connectivity...');
meshfile = fullfile('gmsh', 'channel2.msh'); 
mesh = read_gmsh(meshfile); 
mesh = fix_orientation(mesh);
conn = build_connectivity(mesh);

disp('Evaluating Levelset and Immersed Boundary...');
cx = 0.2;
cy = 0.2;
R = 0.05; % D = 0.1 (50% blockage ratio)
phi = eval_levelset(mesh.nodes, cx, cy, R);

disp('Classifying Elements...');
classification = classify_elements(conn, phi);

disp('Building Ghost Faces...');
faces = build_face_connectivity(conn);
ghost_info = build_ghost_faces(faces, classification);

%% 2. DOF Allocation
disp('Mapping Degrees of Freedom...');
ndpn = 3;
dof = dof_map(mesh, ndpn);
edof = element_dof_map(conn, dof); 

%% 3. Physics & Time Setup
disp('Setting up Transient Physics Parameters...');
Re_D = 100;                        % Physical Reynolds number (U_mean*D/nu)
H = 0.41;                         % Channel height
U_max = 1.5;                      % Peak parabolic inlet velocity
U_mean = (2/3) * U_max;           % = 1.0 m/s
D = 0.1;                          % Cylinder diameter
nu = U_mean * D / Re_D;           % = 0.001 m^2/s
Re = 1 / nu;                      % Code uses 1/Re as viscosity → Re = 1000
f_source = {@(x,y) 0, @(x,y) 0, @(x,y) 0};

dt = 0.005; 
t_start = 0.0;
t_end = 10; 
time_steps = ceil((t_end - t_start) / dt);

% CutFEM Penalties
cutfem_params.alpha_v = 0.05;   % Tuned via Phase 1 pure Stokes
cutfem_params.alpha_p = 0.1;    % Tuned via Phase 1 pure Stokes
cutfem_params.alpha_adv = 0.01; % Tuned via Phase 2 high Advection
cutfem_params.gamma_u = 40; % Higher for higher order or finer meshes

%% 4. Boundary Conditions
disp('Applying Boundary Conditions...');
bc_dofs_base = []; bc_vals_base = [];

% A. Inlet (Dirichlet: u = Parabolic * Ramp, v=0)
inlet_nodes = get_boundary_nodes(mesh, {'inlet'});
inlet_u_dofs = [];
inlet_u_base_vals = [];
for i = 1:length(inlet_nodes)
    y_node = mesh.nodes(inlet_nodes(i), 2);
    u_base = 4 * U_max * y_node * (H - y_node) / H^2;
    inlet_u_dofs = [inlet_u_dofs; dof.node(inlet_nodes(i), 1)]; 
    inlet_u_base_vals = [inlet_u_base_vals; u_base];
    bc_dofs_base = [bc_dofs_base; dof.node(inlet_nodes(i), 2)]; bc_vals_base = [bc_vals_base; 0.0];
end

% B. Walls (Dirichlet: u=0, v=0)
wall_nodes = get_boundary_nodes(mesh, {'walltop', 'wallbottom'});
for i = 1:length(wall_nodes)
    bc_dofs_base = [bc_dofs_base; dof.node(wall_nodes(i), 1)]; bc_vals_base = [bc_vals_base; 0.0];
    bc_dofs_base = [bc_dofs_base; dof.node(wall_nodes(i), 2)]; bc_vals_base = [bc_vals_base; 0.0];
end

% C. Dummy Solid BCs (Dirichlet: u=0, v=0, p=0 for isolated solid nodes)
active_nodes = [];
elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
for t = 1:length(elem_types)
    type = elem_types{t};
    if isfield(classification, type)
        fluid = classification.(type).FULL_FLUID;
        cut = classification.(type).CUT;
        active_elems = conn.(type)([fluid; cut], :);
        active_nodes = [active_nodes; active_elems(:)];
    end
end
active_nodes = unique(active_nodes);
all_nodes = (1:size(mesh.nodes, 1))';
solid_nodes = setdiff(all_nodes, active_nodes);

for i = 1:length(solid_nodes)
    bc_dofs_base = [bc_dofs_base; dof.node(solid_nodes(i), 1); dof.node(solid_nodes(i), 2); dof.node(solid_nodes(i), 3)];
    bc_vals_base = [bc_vals_base; 0.0; 0.0; 0.0];
end

% D. Outlet Neumann (Zero Traction)
if isfield(mesh.boundaries, 'outlet')
    outlet_edge_conn = mesh.boundaries.outlet.edges;
    neumann_bcs = { {outlet_edge_conn, 0.0, 0.0} };
else
    neumann_bcs = {};
end

%% 5. Initialization
disp('Setting initial conditions: u=U_mean, v=0, p=0 (except inside cylinder)');
U_n = set_ic(dof, U_mean, 0, 0);
for i = 1:size(mesh.nodes, 1)
    if sqrt((mesh.nodes(i,1)-cx)^2 + (mesh.nodes(i,2)-cy)^2) <= R
        U_n(dof.node(i,1)) = 0;
        U_n(dof.node(i,2)) = 0;
    end
end

tol_picard = 1e-6;  max_iter_picard = 100;
time_history = zeros(time_steps, 1);
CL_history = zeros(time_steps, 1);
CD_history = zeros(time_steps, 1);

fig_vel = figure('Name', 'u-Velocity & Streamlines', 'Position', [100, 200, 500, 400]);
fig_vvel = figure('Name', 'v-Velocity Field', 'Position', [620, 200, 500, 400]);
fig_pres = figure('Name', 'Pressure Field', 'Position', [1140, 200, 500, 400]);
fig_forces = figure('Name', 'Aerodynamic Forces over Time', 'Position', [100, 650, 1020, 300]);

%% 6. Transient Time-Stepping Loop
disp('Starting CutFEM Transient Simulation...');
tic_gpu = tic;

for t_step = 1:time_steps
    current_time = t_step * dt;
    
    bc_dofs = [inlet_u_dofs; bc_dofs_base];
    bc_vals = [inlet_u_base_vals; bc_vals_base];
    
    fprintf('\n=== Time Step %d / %d (t = %.3f) ===\n', t_step, time_steps, current_time);
    
    U_guess = U_n; 
    
    [U_new, total_iters, ~] = picard_cutfem(mesh, conn, edof, dof, Re, dt, ...
        U_guess, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, ...
        tol_picard, max_iter_picard, phi, classification, ghost_info, cutfem_params);
    
    % --- Aerodynamic Forces ---
    [~, ~, C_D, C_L] = calc_aero_forces_CutFEM(mesh, conn, edof, dof, Re, U_new, ...
        phi, classification, cutfem_params, D, U_mean);
    
    time_history(t_step) = current_time;
    CD_history(t_step) = C_D;
    CL_history(t_step) = C_L;
    
    fprintf('  -> C_D = %.4f | C_L = %.4f\n', C_D, C_L);
    
    if mod(t_step, 5) == 0
        set(0, 'CurrentFigure', fig_vel);
        clf(fig_vel);
        plot_results(mesh, conn, dof, U_new, 1, []);
        hold on;
        theta_plot = linspace(0, 2*pi, 100);
        plot(cx + R*cos(theta_plot), cy + R*sin(theta_plot), 'k-', 'LineWidth', 2);
        hold off;
        title(sprintf('CutFEM u-Velocity | Time: %.2f', current_time));
        
        set(0, 'CurrentFigure', fig_vvel);
        clf(fig_vvel);
        plot_results(mesh, conn, dof, U_new, 2, []);
        hold on; plot(cx + R*cos(theta_plot), cy + R*sin(theta_plot), 'k-', 'LineWidth', 2); hold off;
        title(sprintf('CutFEM v-Velocity | Time: %.2f', current_time));
        
        set(0, 'CurrentFigure', fig_pres);
        clf(fig_pres);
        plot_results(mesh, conn, dof, U_new, 3, []);
        hold on; plot(cx + R*cos(theta_plot), cy + R*sin(theta_plot), 'k-', 'LineWidth', 2); hold off;
        title(sprintf('CutFEM Pressure | Time: %.2f | C_D: %.3f', current_time, C_D));
        
        set(0, 'CurrentFigure', fig_forces);
        subplot(1,2,1);
        plot(time_history(1:t_step), CD_history(1:t_step), 'b-', 'LineWidth', 1.5);
        title('Drag Coefficient (C_D)'); xlabel('Time'); ylabel('C_D'); grid on;
        xlim([0 t_end]);
        
        subplot(1,2,2);
        plot(time_history(1:t_step), CL_history(1:t_step), 'r-', 'LineWidth', 1.5);
        title('Lift Coefficient (C_L)'); xlabel('Time'); ylabel('C_L'); grid on;
        xlim([0 t_end]);
        
        drawnow;
    end
    
    U_n = U_new;
end

%% 7. Final Flow Field Plot
disp('Simulation Complete. Final plots have been dynamically updated.');

%% 8. Post-Processing: Strouhal Number Calculation
disp('Calculating Strouhal Number...');

steady_start = floor(0.3 * time_steps);
t_steady = time_history(steady_start:end);
CL_steady = CL_history(steady_start:end);

Fs = 1 / dt;
L = length(CL_steady);
Y = fft(CL_steady - mean(CL_steady));
P2 = abs(Y / L);
P1 = P2(1:floor(L/2)+1);
P1(2:end-1) = 2 * P1(2:end-1);
frequencies = Fs * (0:(L/2)) / L;

[~, max_idx] = max(P1);
f_s = frequencies(max_idx);
St = (f_s * D) / U_mean;
fprintf('\n>>> Computed Strouhal Number: St = %.4f <<<\n', St);

figure(fig_forces);
subplot(1,2,1);
plot(time_history, CD_history, 'b-', 'LineWidth', 1.5);
title('CutFEM Drag Coefficient (C_D)'); xlabel('Time'); ylabel('C_D'); grid on;

subplot(1,2,2);
plot(time_history, CL_history, 'r-', 'LineWidth', 1.5);
title('CutFEM Lift Coefficient (C_L)'); xlabel('Time'); ylabel('C_L'); grid on;

%% 8. Performance Profiling
wait(gpu_dev);
total_time = toc(tic_gpu);
disp('--- CutFEM Performance Stats ---');
fprintf('Total Simulation Wall Time: %.2f s\n', total_time);
fprintf('Avg Time per Step: %.3f s\n', total_time / time_steps);
