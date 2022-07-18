function yOut = GPU_Euler(t,h,yOut,yOld,dy,gJ,cV) %#codegen

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

% Update state variables with Euler's method
rep = length(t);
for idx = 1:rep
    
    % calculating new states
    yNew = yOld + h * Model_1D_GPU(t(idx),yOld,dy,gJ,cV);    
    yOld = yNew; % the update state is the initial condition of the next teration
    yOut(:,:,idx) = yNew; % save every step in an output vector    
    
end
