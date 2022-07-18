function yOut = Severi_GPU_2D_Euler(t,h,yOut,yOld,dy,gJ,cV, vI) %#codegen

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

% Update state variables with Euler's method
rep = length(t);
for idx = 1:rep
    
    % calculating new states
    yNew = yOld + h * SeveriModel_2D_GPU(t(idx),yOld,dy,gJ,cV, vI);    
    % the update state is the initial condition of the next iteration
    yOld = yNew; 
    % save every step in an output vector 
    yOut(:,:,:,idx) = yNew; 

end