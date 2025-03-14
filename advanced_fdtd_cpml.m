% Advanced 2D FDTD Implementation with CPML and TF/SF Gaussian Source Injection
clear; clc;

%% Constants
mo = 4 * pi * 1e-7; % Permeability of free space [H/m]
co = 299792458;     % Speed of light in vacuum [m/s]
eo = 1 / (mo * co^2); % Permittivity of free space [F/m]
Zo = sqrt(mo / eo); % Impedance of free space [Ohms]
eta0 = Zo;          % Intrinsic impedance of free space

%% Model Parameters
NX = 400;           % Domain size in X (cells)
NY = 400;           % Domain size in Y (cells)
NPML = 10;          % Number of PML layers
Iter = 5000;         % Number of iterations

dx = 0.01;          % Spatial step in x [m]
dy = 0.01;          % Spatial step in y [m]
dt = 1 / (co * sqrt(1 / dx^2 + 1 / dy^2)); % CFL stability condition

disp(['Velocity of Light [m/s]: ', num2str(co)]);
disp(['Permittivity of Free Space [F/m]: ', num2str(eo)]);
disp(['Permeability of Free Space [H/m]: ', num2str(mo)]);
disp(['Impedance of Free Space [Ohms]: ', num2str(Zo)]);

disp(['FDTD Model Space in Cells: ', num2str(NX), 'x', num2str(NY)]);
disp(['Number of Iterations: ', num2str(Iter)]);
disp(['dx = ', num2str(dx), ', dy = ', num2str(dy), ', dt = ', num2str(dt)]);

%% Field Arrays
Ez = zeros(NX, NY); % TMz field array
Hx = zeros(NX, NY+1); % Hx field array
Hy = zeros(NX+1, NY); % Hy field array

%% CPML Parameters
PML_R0 = 1e-5;      % Reflection coefficient
PML_m = 3;          % PML grading exponent
PML_max_sigma = (0.8 * (PML_m + 1)) / (eta0 * dx);
sigmax = PML_max_sigma * ((0:NPML-1) / NPML).^PML_m;
sigmay = PML_max_sigma * ((0:NPML-1) / NPML).^PML_m;

% Initialize PML arrays
psi_Ex_xlo = zeros(NPML, NY);
psi_Ex_xhi = zeros(NPML, NY);
psi_Ey_ylo = zeros(NX, NPML);
psi_Ey_yhi = zeros(NX, NPML);

%% TF/SF Parameters
src_Sx = NX / 2;    % Source location (X)
src_Sy = NY / 2;    % Source location (Y)

%% Material Properties
ER = ones(NX, NY) * eo; % Relative permittivity
SIGMA = zeros(NX, NY);  % Conductivity

% Add a dielectric cylinder
Radius = 20;
CenterX = NX / 2;
CenterY = NY / 2;
relative_permittivity = 4.0;
sigma = 0.1;

for i = CenterX - Radius:CenterX + Radius
    for j = CenterY - Radius:CenterY + Radius
        if sqrt((i - CenterX)^2 + (j - CenterY)^2) <= Radius
            ER(i, j) = relative_permittivity * eo;
            SIGMA(i, j) = sigma;
        end
    end
end

%% Precompute Update Coefficients
DA = (2 * ER - SIGMA * dt) ./ (2 * ER + SIGMA * dt);
CA = (2 * dt ./ (2 * ER + SIGMA * dt));

DB = 1;
CB = dt / mo;

%% Gaussian Source Definition
tw = 26.53e-6; % Gaussian pulse width
to = 4 * tw;    % Pulse center time
src = @(t) exp(-((t - to) / tw)^2); % Gaussian function

%% Main FDTD Loop
tic;
for n = 1:Iter
    %% Update Magnetic Fields (Hx, Hy)
    for i = 1:NX
        for j = 2:NY
            Hx(i, j) = DB * Hx(i, j) - CB * (Ez(i, j) - Ez(i, j - 1)) / dy;
        end
    end

    for i = 2:NX
        for j = 1:NY
            Hy(i, j) = DB * Hy(i, j) + CB * (Ez(i, j) - Ez(i - 1, j)) / dx;
        end
    end

    %% Apply CPML Updates
    for p = 1:NPML
        % X-lo boundary
        psi_Ex_xlo(p, :) = DA(p, :) .* psi_Ex_xlo(p, :) ...
            + CA(p, :) .* (Ez(p, :) - Ez(p + 1, :)) / dx;
        Ez(p, :) = Ez(p, :) + psi_Ex_xlo(p, :);

        % X-hi boundary
        psi_Ex_xhi(p, :) = DA(NX - p + 1, :) .* psi_Ex_xhi(p, :) ...
            + CA(NX - p + 1, :) .* (Ez(NX - p, :) - Ez(NX - p - 1, :)) / dx;
        Ez(NX - p, :) = Ez(NX - p, :) + psi_Ex_xhi(p, :);

        % Y-lo boundary
        psi_Ey_ylo(:, p) = DA(:, p) .* psi_Ey_ylo(:, p) ...
            + CA(:, p) .* (Ez(:, p) - Ez(:, p + 1)) / dy;
        Ez(:, p) = Ez(:, p) + psi_Ey_ylo(:, p);

        % Y-hi boundary
        psi_Ey_yhi(:, p) = DA(:, NY - p + 1) .* psi_Ey_yhi(:, p) ...
            + CA(:, NY - p + 1) .* (Ez(:, NY - p) - Ez(:, NY - p - 1)) / dy;
        Ez(:, NY - p) = Ez(:, NY - p) + psi_Ey_yhi(:, p);
    end

    %% Update Electric Field (Ez)
    for i = 2:NX-1
        for j = 2:NY-1
            Ez(i, j) = DA(i, j) * Ez(i, j) ...
                + CA(i, j) * ((Hy(i + 1, j) - Hy(i, j)) / dx ...
                - (Hx(i, j + 1) - Hx(i, j)) / dy);
        end
    end

    %% Inject Gaussian Source at TF/SF Interface
    t = n * dt; % Current simulation time
    Ez(src_Sx, src_Sy) = Ez(src_Sx, src_Sy) - CA(src_Sx, src_Sy) * src(t) / (dx * dy);

    %% Visualization
    if mod(n, 10) == 0
        figure(1);
        clf;
        subplot(1, 3, 1);
        imagesc(Ez); axis square; colorbar; title('Ez');
        subplot(1, 3, 2);
        imagesc(Hx); axis square; colorbar; title('Hx');
        subplot(1, 3, 3);
        imagesc(Hy); axis square; colorbar; title('Hy');
        drawnow;
    end
end
toc;

disp('Simulation Complete!');
