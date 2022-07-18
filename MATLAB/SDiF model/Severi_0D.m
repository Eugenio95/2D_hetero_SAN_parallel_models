close all
clear
clc

addpath C:\Users\eucci\OneDrive\Desktop\Dottorato\SAN_Project\Severi_2012
% Initial Conditions
Y0 = [7.86181717518e-8, 1.7340201253e-7, 0.912317231017262, 0.211148145512825, 0.0373817991524254, 0.054381370046, 0.299624275428735, 0.0180519400676086, 0.281244308217086, 0.501049376634, 0.316762674605, 1.05386465080816, 1.0e-5, 1.0e-5, 0.0, 0.0, -52.0, 7.5, 0.0, 0.697998543259722, 0.497133507285601, 0.0, 0.0, 0.0, 0.0990510403258968, 0.322999177802891, 0.705410877258545, 0.0, 1.3676940140066e-5, 0.440131579215766, 0.181334538702451, 0.506139850982478, 0.0144605370597924];
tSpan = [0 40]; %s

% Waitbar
options = odeset('OutputFcn',@odewbar);

%% Solve with ODE15s
clock = tic;
[t,y] = ode15s(@SeveriOriginal_Model_2012, tSpan, Y0, options);
toc(clock)

% Plot
figure
plot(t,y(:,17)) 
title('SAN Action Potential')%, ','R= ',num2str(rGap/1e6),' MOhm']), xlabel('t [s]'), ylabel('V [mV]')
