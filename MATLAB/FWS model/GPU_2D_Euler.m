function yOut = GPU_2D_Euler(t, h, yOut, yOld, dy, gJ, cV, underSamp) %#codegen

% Add kernelfun pragma to trigger kernel creation
coder.gpu.kernelfun;

% Update state variables with Euler's method
rep = length(t);
for idx = 1:rep
    
    % calculating new states
    yOld = yOld + h * Model_2D_GPU(t(idx), yOld, dy, gJ, cV);    
    
    if mod(idx,underSamp) == 0
        k = idx/underSamp;
        yOut(:,:,:,k) = yOld;
    end

end