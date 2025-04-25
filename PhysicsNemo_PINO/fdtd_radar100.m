clear all;
close all;

% Constants
mo = 400 * pi * 1e-9;
co = 2.997925e8;
eo = 1.0 / (mo * (co)^2);
Zo = sqrt(mo / eo);

% Model params
NX = 100; % Domain size in X in cells
NY = 100; % Domain size in Y in cells
Iter = 207; % Number of iterations
dx = 0.001; % Spatial step in x
dy = 0.001; % Spatial step in y
dt = 1 / (co * sqrt(1.0 / (dx^2) + 1.0 / (dy^2))); % CFL stability condition

% Number of simulations
N_simulations = 100;

% Output directory
output_dir = 'FDTD_simulations';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Randomization ranges
source_x_range = [10, NX-10];
source_y_range = [10, NY-10];
center_x_range = [25, NX-25];
center_y_range = [25, NY-25];
radius_range = [5, 15];
permittivity_range = [2.0, 6.0];
sigma_range = [0.05, 0.15];

% Source definition
tw = 26.53e-12;
to = 4 * tw;
src = zeros(1, Iter+1);
for i = 1:Iter+1
    t = i * dt;
    src(i) = -2.0 * ((t - to) / tw) * exp(-((t - to) / tw)^2);
end

% Loop over simulations
for sim_idx = 1:N_simulations
    % Set random seed for reproducibility
    rng(sim_idx-1);

    % Randomize parameters
    Sx = randi([source_x_range(1), source_x_range(2)]);
    Sy = randi([source_y_range(1), source_y_range(2)]);
    CenterX = randi([center_x_range(1), center_x_range(2)]);
    CenterY = randi([center_y_range(1), center_y_range(2)]);
    Radius = randi([radius_range(1), radius_range(2)]);
    relative_permittivity = permittivity_range(1) + (permittivity_range(2) - permittivity_range(1)) * rand();
    sigma = sigma_range(1) + (sigma_range(2) - sigma_range(1)) * rand();

    % Initialize fields
    Ez = zeros(NX, NY);
    Hx = zeros(NX, NY+1);
    Hy = zeros(NX+1, NY);

    % Initialize material properties
    ER = ones(NX, NY) * eo;
    SIGMA = zeros(NX, NY);

    % Build cylinder
    for i = CenterX-Radius-1:CenterX+Radius+1
        for j = CenterY-Radius-1:CenterY+Radius+1
            if i >= 1 && i <= NX && j >= 1 && j <= NY
                if sqrt((i-CenterX)^2 + (j-CenterY)^2) <= Radius
                    ER(i,j) = relative_permittivity * eo;
                    SIGMA(i,j) = sigma;
                end
            end
        end
    end

    % Coefficients
    DA = (2 * ER - SIGMA * dt) ./ (2 * ER + SIGMA * dt);
    CA = (2 * dt ./ (2 * ER + SIGMA * dt));
    DB = 1;
    CB = dt / mo;

    % Output arrays for all time steps
    Ez_out = zeros(Iter+1, NX, NY);
    Hx_out = zeros(Iter+1, NX, NY+1);
    Hy_out = zeros(Iter+1, NX+1, NY);

    % Time stepping
    ind = 1;
    time = 0;
    while ind < Iter
        % Update Hx
        for i = 1:NX
            for j = 2:NY
                Hx(i,j) = DB * Hx(i,j) - CB * (Ez(i,j) - Ez(i,j-1)) / dy;
            end
        end

        % Update Hy
        for i = 2:NX
            for j = 1:NY
                Hy(i,j) = DB * Hy(i,j) + CB * (Ez(i,j) - Ez(i-1,j)) / dx;
            end
        end

        time = time + dt * 0.5;

        % Update Ez
        for i = 1:NX
            for j = 1:NY
                Ez(i,j) = DA(i,j) * Ez(i,j) + CA(i,j) * (Hy(i+1,j) - Hy(i,j)) / dx - ...
                          CA(i,j) * (Hx(i,j+1) - Hx(i,j)) / dy;
            end
        end

        % Adjust Ez at source
        Ez(Sx,Sy) = Ez(Sx,Sy) - CA(Sx,Sy) * src(ind) * (1 / (dx * dy));

        time = time + dt * 0.5;

        % Store fields
        Ez_out(ind,:,:) = Ez;
        Hx_out(ind,:,:) = Hx;
        Hy_out(ind,:,:) = Hy;

        ind = ind + 1;
    end

    % Save results for this simulation
    output_file = fullfile(output_dir, sprintf('results_%04d.mat', sim_idx));
    save(output_file, 'Ez_out', 'Hx_out', 'Hy_out', 'Sx', 'Sy', ...
         'CenterX', 'CenterY', 'Radius', 'relative_permittivity', 'sigma');

    % Display progress
    disp(['Completed simulation ' num2str(sim_idx) ' of ' num2str(N_simulations)]);
end

disp('All simulations completed!');