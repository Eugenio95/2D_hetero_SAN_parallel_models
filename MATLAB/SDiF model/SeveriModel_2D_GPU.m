function dV = SeveriModel_2D_GPU(time,V,dV,gJ,constVars, vI) %#codegen

VmIndex = vI;

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

% Update state variables for every cell
for i = 1:size(V,2)
    for j = 1:size(V,3)
        dV(:,i,j) = GPU_SeveriModel2012(time,V(:,i,j),constVars(i,j));
    end
end
% Compute difference in Membrane Potential
Voriz = V(VmIndex,:,[2:end, end]) - 2 * V(VmIndex,:,:) + V(VmIndex,:,[1, 1:end-1]); 
Vvert = V(VmIndex,[2:end, end],:) - 2 * V(VmIndex,:,:) + V(VmIndex,[1, 1:end-1],:);
Vnet= Voriz + Vvert;

% Add propagation current : dY(index,1) = -i_tot/C + (Vj-Vi)/(Rm*Cm);
dV(VmIndex,:,:) = squeeze(dV(VmIndex,:,:)) + squeeze(Vnet)./gJ;

end