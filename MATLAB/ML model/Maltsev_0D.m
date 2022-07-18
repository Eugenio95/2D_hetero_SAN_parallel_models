close all
clear
clc

% Initial Conditions
Y0 = [1.0, 0.0, -65.0, 0.042, 0.089, 0.032, 0.02, 0.22, 0.69, 0.029, 1.35, 0.000223, 0.0001, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.1e-6, 3.4e-6, 0.7499955, 0.25];
tSpan = [0 4e4];
MaltsevConstants
% Waitbar
options = odeset('OutputFcn',@odewbar);

%% Solve with ODE15s
clock = tic;
[t,y] = ode15s(@MaltsevOriginal_Model_2009, tSpan, Y0, options);
toc(clock)
t = t/1e3; %from ms to s

% Plot
figure
plot(t,y(:,3)) 
title('SAN Action Potential')%, ','R= ',num2str(rGap/1e6),' MOhm']), xlabel('t [s]'), ylabel('V [mV]')
