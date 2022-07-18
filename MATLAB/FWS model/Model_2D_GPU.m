function dV = Model_2D_GPU(time,V,dV,gJ,constVars) %#codegen

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

% Update state variables for every cell
for i = 1:size(V,2)
    for j = 1:size(V,3)
        % Every cell ha its struct of constants
        dV(:,i,j) = GPU_HumanSAN_Fabbri_Fantini_Wilders_Severi_2017(time,V(:,i,j),constVars(i,j));
    end
end

%%% Compute difference in Membrane Potential

% V_u = V(15,:,[2:end, end]) - V(15,:,:);
% V_d = V(15,:,[1, 1:end-1]) - V(15,:,:);
% V_l = V(15,[2:end, end],:) - V(15,:,:);
% V_r = V(15,[1, 1:end-1],:) - V(15,:,:);
% Vnet = V_u + V_d + V_l + V_r; 

% Vqq = V(15,[2:end, end],[2:end, end]) - 2 * V(15,:,:) + V(15,[1,1:end-1],[1,1:end-1]);
% Vmist = (V(15,[2:end, end],[2:end, end]) - V(15,[1,1:end-1],[1,1:end-1]) ...
%         + V(15,[2:end, end], [1,1:end-1]) - V(15,[1,1:end-1],[2:end, end]) ) / 4;

Voriz = V(15,:,[2:end, end]) - 2 * V(15,:,:) + V(15,:,[1, 1:end-1]); % == V_u + V_d
Vvert = V(15,[2:end, end],:) - 2 * V(15,:,:) + V(15,[1, 1:end-1],:);
Vnet  = squeeze(Voriz + Vvert);

% Add propagation current : dY(15,1) = -i_tot/C + (Vj-Vi)/(Rm*Cm);
dV(15,:,:) = squeeze(dV(15,:,:)) + Vnet./gJ;

end

