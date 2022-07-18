clear 
close all
clc

addpath Malt_Results

load Maltsev50x50_20s_100Gohm_var-0.4.mat


%% 
n = 33;
pot = squeeze(AP(n,20,:));

figure
plot(tt,pot)
legend('AP')