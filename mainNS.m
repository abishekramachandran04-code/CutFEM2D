% mainNS.m (Transient Navier-Stokes: Flow Past a Cylinder)
clear; clc; close all;
addpath('mesh', 'fem', 'physics', 'utils', 'bc', 'solvers', 'post');

%% 0. GPU Initialization
gpu_dev = gpu_init();

%% 1. Mesh Processing
disp('Reading Mesh and Building Connectivity...');
meshfile = fullfile('gmsh', 'NSmesh_Q4.msh'); 
mesh = read_gmsh(meshfile); 
mesh = fix_orientation(mesh);
conn = build_connectivity(mesh);

%% 2. DOF Allocation
disp('Mapping Degrees of Freedom...');
ndpn = 3;
dof = dof_map(mesh, ndpn);
edof = element_dof_map(conn, dof); 

%% 3. Physics & Time Setup
disp('Setting up Transient Physics Parameters...');
Re_D = 100;                        % Physical Reynolds number (U_mean*D/nu)
H = 0.41;                          % Channel height
U_max = 1.5;                       % Peak parabolic inlet velocity
U_mean = (2/3) * U_max;            % = 1.0 m/s
D = 0.1;                           % Cylinder diameter
nu = U_mean * D / Re_D;            % = 0.001 m^2/s
Re = 1 / nu;                       % Code uses 1/Re as viscosity → Re = 1000
f_source = {@(x,y) 0, @(x,y) 0, @(x,y) 0};

dt = 0.005; 
t_start = 0.0;
t_end = 10;
time_steps = ceil((t_end - t_start) / dt);

%% 4. Boundary Conditions
disp('Applying Boundary Conditions...');
bc_dofs = [];
bc_vals = [];

% A. Inlet: PARABOLIC velocity profile u(y) = 4*U_max*y*(H-y)/H^2
inlet_nodes = get_boundary_nodes(mesh, {'inlet'});
for i = 1:length(inlet_nodes)
    y_node = mesh.nodes(inlet_nodes(i), 2);
    u_inlet = 4 * U_max * y_node * (H - y_node) / H^2;
    bc_dofs = [bc_dofs; dof.node(inlet_nodes(i), 1)]; bc_vals = [bc_vals; u_inlet];
    bc_dofs = [bc_dofs; dof.node(inlet_nodes(i), 2)]; bc_vals = [bc_vals; 0.0];
end

% B. Top and Bottom Walls: NO-SLIP (u=0, v=0) — required for DFG benchmark
wall_nodes = get_boundary_nodes(mesh, {'walltop', 'wallbottom'});
for i = 1:length(wall_nodes)
    bc_dofs = [bc_dofs; dof.node(wall_nodes(i), 1)]; bc_vals = [bc_vals; 0.0];
    bc_dofs = [bc_dofs; dof.node(wall_nodes(i), 2)]; bc_vals = [bc_vals; 0.0];
end

cylinder_nodes = get_boundary_nodes(mesh, {'wallcylinder'});
for i = 1:length(cylinder_nodes)
    bc_dofs = [bc_dofs; dof.node(cylinder_nodes(i), 1)]; bc_vals = [bc_vals; 0.0];
    bc_dofs = [bc_dofs; dof.node(cylinder_nodes(i), 2)]; bc_vals = [bc_vals; 0.0];
end

outlet_edge_conn = mesh.boundaries.outlet.edges;
P_nd = 0.0;
tau_nd = 0.0;
neumann_bcs = { {outlet_edge_conn, P_nd, tau_nd} };

%[~, min_idx] = min(mesh.nodes(inlet_nodes, 2));
%bottom_left_node = inlet_nodes(min_idx);
%bc_dofs = [bc_dofs; dof.node(bottom_left_node, 3)];
%bc_vals = [bc_vals; 0.0];

%% 5. Initialization
tol_picard = 1e-6;  max_iter_picard = 30;
tol_nr = 1e-6;      max_iter_nr = 10;

U_n = set_ic(dof, U_mean, 0.0, 0.0);

time_history = zeros(time_steps, 1);
CL_history = zeros(time_steps, 1);
CD_history = zeros(time_steps, 1);

fig_vel = figure('Name', 'u-Velocity & Streamlines', 'Position', [100, 200, 500, 400]);
fig_vvel = figure('Name', 'v-Velocity Field', 'Position', [620, 200, 500, 400]);
fig_pres = figure('Name', 'Pressure Field', 'Position', [1140, 200, 500, 400]);
fig_forces = figure('Name', 'Aerodynamic Forces over Time', 'Position', [100, 650, 1020, 300]);

%% 6. The Transient Time-Stepping Loop
disp('Starting Transient Simulation...');
tic_gpu = tic;

for t_step = 1:time_steps
    current_time = t_step * dt;
    fprintf('\n=== Time Step %d / %d (t = %.3f) ===\n', t_step, time_steps, current_time);
    
    U_guess = U_n; 
    
    [U_new, total_iters, ~] = picard(mesh, conn, edof, dof, Re, dt, ...
        U_guess, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, ...
        tol_picard, max_iter_picard);
    
    [~, ~, C_D, C_L] = calc_aero_forces(mesh, conn, edof, dof, Re, dt, U_new, U_n, f_source, cylinder_nodes, D, U_mean);
    
    time_history(t_step) = current_time;
    CL_history(t_step) = C_L;
    CD_history(t_step) = C_D;
    
    fprintf('  -> C_D = %.4f | C_L = %.4f\n', C_D, C_L);
    
    if mod(t_step, 5) == 0
        set(0, 'CurrentFigure', fig_vel);
        clf(fig_vel);
        plot_results(mesh, conn, dof, U_new, 1, []);
        hold on;
        plot_streamlines(mesh, dof, U_new);
        hold off;
        title(sprintf('u-Velocity & Streamlines | Time: %.2f | C_L: %.3f', current_time, C_L));
        
        set(0, 'CurrentFigure', fig_vvel);
        clf(fig_vvel);
        plot_results(mesh, conn, dof, U_new, 2, []);
        title(sprintf('v-Velocity Field | Time: %.2f ', current_time));
        
        set(0, 'CurrentFigure', fig_pres);
        clf(fig_pres);
        plot_results(mesh, conn, dof, U_new, 3, []);
        title(sprintf('Pressure Field | Time: %.2f | C_D: %.3f', current_time, C_D));
        
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

%% 7. Post-Processing: Strouhal Number Calculation
disp('Simulation Complete. Calculating Strouhal Number...');

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
title('Drag Coefficient (C_D)'); xlabel('Time'); ylabel('C_D'); grid on;

subplot(1,2,2);
plot(time_history, CL_history, 'r-', 'LineWidth', 1.5);
title('Lift Coefficient (C_L)'); xlabel('Time'); ylabel('C_L'); grid on;

%% 8. Performance Profiling
wait(gpu_dev);
total_time = toc(tic_gpu);
disp('--- Performance Stats ---');
fprintf('Total Simulation Wall Time: %.2f s\n', total_time);
fprintf('Avg Time per Step: %.3f s\n', total_time / time_steps);
fprintf('GPU Device: %s\n', gpu_dev.Name);
fprintf('GPU Memory Used: %.1f / %.1f MB\n', ...
    (gpu_dev.TotalMemory - gpu_dev.AvailableMemory)/1e6, gpu_dev.TotalMemory/1e6);