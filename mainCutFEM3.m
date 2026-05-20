% mainCutFEM3.m (Dynamic CutFEM: Oscillating Cylinder Benchmark)
clear; clc; close all;
addpath('mesh', 'fem', 'physics', 'utils', 'bc', 'solvers', 'post');

%% 0. GPU Initialization
gpu_dev = gpu_init();

%% 1. Mesh Processing
disp('Reading Mesh and Building Connectivity...');
meshfile = fullfile('gmsh', 'channel2.msh'); 
mesh = read_gmsh(meshfile); 
mesh = fix_orientation(mesh);
conn = build_connectivity(mesh);
faces = build_face_connectivity(conn);

%% 2. DOF Allocation
disp('Mapping Degrees of Freedom...');
ndpn = 3;
dof = dof_map(mesh, ndpn);
edof = element_dof_map(conn, dof); 

%% 3. Physics & Time Setup
disp('Setting up Transient Physics Parameters...');
Re_D = 100;                        % Reynolds number
H = 0.41;                         % Channel height
U_max = 1.5;                      % Peak parabolic inlet velocity
U_mean = (2/3) * U_max;           % = 1.0 m/s
D = 0.1; R = D/2;                 % Cylinder geometry
nu = U_mean * D / Re_D;
Re = 1 / nu;
f_source = {@(x,y) 0, @(x,y) 0, @(x,y) 0};

% Benchmark Oscillation Parameters
% Transverse oscillation: y(t) = y_0 + A*sin(2*pi*f_e*t)
A = 0.2 * D;                      % Amplitude = 0.02
% DFG 2D-2 natural shedding has St ~ 0.30. 
% Therefore f_0 = St * U_mean / D = 0.30 * 1.0 / 0.1 = 3.0 Hz
f_e = 3.0;                        % Excitation frequency matching the actual channel lock-in
omega = 2 * pi * f_e;
cx_base = 0.2;
cy_base = 0.2;

dt = 0.005; 
t_start = 0.0;
t_end = 10; 
time_steps = ceil((t_end - t_start) / dt);

% CutFEM Penalties
cutfem_params.alpha_v = 0.05;
cutfem_params.alpha_p = 0.1;
cutfem_params.alpha_adv = 0.01;
cutfem_params.gamma_u = 40;

%% 4. Static Boundary Conditions (Inlet, Walls, Outlet)
disp('Applying Static Boundary Conditions...');
bc_dofs_base = []; bc_vals_base = [];

% A. Inlet (Dirichlet: u = Parabolic, v=0)
inlet_nodes = get_boundary_nodes(mesh, {'inlet'});
inlet_u_dofs = []; inlet_u_base_vals = [];
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

% C. Outlet Neumann (Zero Traction)
if isfield(mesh.boundaries, 'outlet')
    neumann_bcs = { {mesh.boundaries.outlet.edges, 0.0, 0.0} };
else
    neumann_bcs = {};
end

%% 5. Initialization
disp('Setting initial conditions...');
U_n = set_ic(dof, U_mean, 0, 0);

% Initial cylinder position and velocity
cy_0 = cy_base + A * sin(omega * t_start);
v_cyl_0 = A * omega * cos(omega * t_start);
phi_0 = eval_levelset(mesh.nodes, cx_base, cy_0, R);

for i = 1:size(mesh.nodes, 1)
    if phi_0(i) <= 0
        U_n(dof.node(i,1)) = 0; 
        U_n(dof.node(i,2)) = v_cyl_0;
    end
end

tol_picard = 1e-6;  max_iter_picard = 100;
time_history = zeros(time_steps, 1);
CL_history = zeros(time_steps, 1);
CD_history = zeros(time_steps, 1);
cy_history = zeros(time_steps, 1);

fig_vel = figure('Name', 'u-Velocity Field', 'Position', [100, 200, 500, 400]);
fig_vvel = figure('Name', 'v-Velocity Field', 'Position', [620, 200, 500, 400]);
fig_pres = figure('Name', 'Pressure Field', 'Position', [1140, 200, 500, 400]);
fig_forces = figure('Name', 'Aerodynamic Forces over Time', 'Position', [100, 650, 1020, 300]);

%% 6. Transient Time-Stepping Loop
disp('Starting Dynamic CutFEM Transient Simulation (Moving Cylinder)...');

% Create output directory for ParaView files
out_dir = 'vtk_output';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
fprintf('Exporting VTK files to ./%s/\n', out_dir);

tic_gpu = tic;

elem_types = {'tri3', 'quad4', 'tri6', 'quad9'};
all_nodes = (1:size(mesh.nodes, 1))';

for t_step = 1:time_steps
    current_time = t_step * dt;
    
    % --- A. Dynamic Geometry Update ---
    cy_t = cy_base + A * sin(omega * current_time);
    v_cyl = A * omega * cos(omega * current_time);
    
    phi = eval_levelset(mesh.nodes, cx_base, cy_t, R);
    classification = classify_elements(conn, phi);
    ghost_info = build_ghost_faces(faces, classification);
    
    % Nitsche BC parameter update
    cutfem_params.uD = 0.0;
    cutfem_params.vD = v_cyl;
    
    % --- B. Dynamic Dummy Solid BCs ---
    % Find all elements that are strictly fluid or cut
    active_nodes = [];
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
    solid_nodes = setdiff(all_nodes, active_nodes);
    
    % Preallocate dynamic dummy Dirichlet BCs
    dynamic_bc_dofs = zeros(length(solid_nodes)*3, 1);
    dynamic_bc_vals = zeros(length(solid_nodes)*3, 1);
    
    idx = 1;
    for i = 1:length(solid_nodes)
        sn = solid_nodes(i);
        dynamic_bc_dofs(idx:idx+2) = [dof.node(sn, 1); dof.node(sn, 2); dof.node(sn, 3)];
        
        % By forcing the deeply buried solid nodes to the instantaneous cylinder velocity (v_cyl),
        % any nodes that are "uncovered" as the cylinder moves will have a near-perfect 
        % initial guess for the fluid boundary layer velocity!
        dynamic_bc_vals(idx:idx+2) = [0.0; v_cyl; 0.0];
        idx = idx + 3;
    end
    
    bc_dofs = [inlet_u_dofs; bc_dofs_base; dynamic_bc_dofs];
    bc_vals = [inlet_u_base_vals; bc_vals_base; dynamic_bc_vals];
    
    % --- C. Picard Solver ---
    fprintf('\n=== Time Step %d / %d (t = %.3f) | cy = %.4f | vc = %.4f ===\n', t_step, time_steps, current_time, cy_t, v_cyl);
    
    U_guess = U_n; 
    
    [U_new, total_iters, ~] = picard_cutfem(mesh, conn, edof, dof, Re, dt, ...
        U_guess, U_n, f_source, bc_dofs, bc_vals, neumann_bcs, ...
        tol_picard, max_iter_picard, phi, classification, ghost_info, cutfem_params);
        
    % --- D. Aerodynamic Forces ---
    [~, ~, C_D, C_L] = calc_aero_forces_CutFEM(mesh, conn, edof, dof, Re, U_new, ...
        phi, classification, cutfem_params, D, U_mean);
        
    time_history(t_step) = current_time;
    CD_history(t_step) = C_D;
    CL_history(t_step) = C_L;
    cy_history(t_step) = cy_t;
    
    fprintf('  -> C_D = %.4f | C_L = %.4f\n', C_D, C_L);
    
    % --- E. Export to ParaView (VTK) ---
    % Instead of slow MATLAB plotting, we export the exact velocity, pressure, 
    % and level-set field to a VTK file at every single time step.
    vtk_filename = fullfile(out_dir, sprintf('cutfem_step_%04d.vtk', t_step));
    export_vtk(vtk_filename, mesh, conn, dof, U_new, phi);
    
    % --- F. Live Aerodynamic Forces Plot ---
    % We still plot the forces live in MATLAB so you can monitor the lock-in!
    if mod(t_step, 5) == 0
        set(0, 'CurrentFigure', fig_forces);
        
        subplot(1,2,1);
        yyaxis left; plot(time_history(1:t_step), CD_history(1:t_step), 'b-', 'LineWidth', 1.5); ylabel('C_D');
        yyaxis right; plot(time_history(1:t_step), cy_history(1:t_step), 'k--', 'LineWidth', 1); ylabel('Cyl y-Pos');
        title('Drag Coefficient (C_D)'); xlabel('Time'); grid on; xlim([0 t_end]);
        
        subplot(1,2,2);
        yyaxis left; plot(time_history(1:t_step), CL_history(1:t_step), 'r-', 'LineWidth', 1.5); ylabel('C_L');
        yyaxis right; plot(time_history(1:t_step), cy_history(1:t_step), 'k--', 'LineWidth', 1); ylabel('Cyl y-Pos');
        title('Lift Coefficient (C_L) vs Oscillation'); xlabel('Time'); grid on; xlim([0 t_end]);
        
        drawnow;
    end
    
    U_n = U_new;
end

%% 7. Post-Processing: Strouhal Number Calculation
disp('Calculating Strouhal Number...');

steady_start = floor(0.3 * time_steps); % Ignore startup transients
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
fprintf('\n>>> Computed FFT Peak Frequency: f = %.4f Hz <<<\n', f_s);
fprintf('>>> Computed Strouhal Number: St = %.4f <<<\n\n', St);

%% 8. Post-Processing Stats
total_time = toc(tic_gpu);
disp('--- Dynamic CutFEM Performance Stats ---');
fprintf('Total Simulation Wall Time: %.2f s\n', total_time);
fprintf('Avg Time per Step: %.3f s\n', total_time / time_steps);
