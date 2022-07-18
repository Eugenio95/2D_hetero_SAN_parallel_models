close all
clear
clc

% Reset th GPU
g = gpuDevice(1);

% Set C++ compiler path
setenv('BASH_ENV','~/.bash_profile');
p = getenv('PATH');
setenv('PATH', [p ':/usr/local/cuda-10.1/bin'])

%% Simulation parameters
nCells = 50;
mCells = 50;

tSpan = [0 10]; % simulate 10 ms every iteration
rep = 2000; % do n iteration: rep*tSpan = total simulation time

fixedStep = 1e-2; % integration step
t = 0:fixedStep:tSpan(2)-fixedStep;
underSamp = 100; % keep 1 sample every 100 (about 1 ms of sampling step having fixedStep 1.25e-5 s)

VmIndex = 3; % Membrane potential is the 15th variable of state vector

gpuRes = 10; % Reset the GPU (clear its memory every 10 iteration, otherwise out of stak error)

% Gap junction
Cm = 32*1e-12; % pF

rGap = [10, 100, 1000, 10000, inf]*1e9; % MOhm
sigma = 0.3; %[0.4, 0.5];  %[0.1, 0.2, 0.3, 0.4, 0.5]; % n% variability (standard deviation)
flagFinalCond = 0;
pCal_scal = 0.5; % [1, 0.8, 0.5]

for kk = 1:length(sigma)
    for jj = 1:length(rGap)
        
        gapJunct = ones(nCells,mCells)*Cm*rGap(jj);
        
        %% Initial conditions
%         load Malt_initSamp.mat
        stateVectorDim = 29; %size(Malt_y_start,2);
        
        % % % % %     % Sample the y_start data file randomly to the the initial conditions of the cells
        % % % % %     initCondSeed = RandStream('mlfg6331_64');
        % % % % %     startPos = datasample(initCondSeed,1:length(Malt_y_start),nCells*mCells); % s, always keeps the same random sample
        % % % % %     %%% SAME INIT CONDS
        % % % % %     % startPos(2:end) = startPos(1);
        % % % % %     % numSim = 3;
        % % % % %     %%%
        % % % % %     % yOld = y_start(startPos,:,:)';
        % % % % %     yOld = reshape(Malt_y_start(startPos,:,:)',stateVectorDim,nCells,mCells);
        
        if flagFinalCond == 1 && rGap(jj) == inf
            load Malt_Init_Cond
            yOld = yStart;
        else
            initCondFile = ['initCondMaltsev-', num2str(sigma(kk)), '.mat'];
            load(initCondFile)
            yOld = finalCond;
        end
%         yOld = round(Malt_InitCond, 5, 'significant');
        % Transfer data to the GPU
        yOld = gpuArray(yOld);
        
        %% Variability
        MaltsevConstants % Call the struct containing the costants of the model
        
        standardConductances = [constStr.g_sus, constStr.g_to, constStr.g_CaL, constStr.g_CaT, constStr.g_Kr, ...
            constStr.g_Ks, constStr.g_b_Ca, constStr.g_b_Na, constStr.g_if, constStr.g_st...
            constStr.i_NaK_max, constStr.kNaCa];
        heterConductances = zeros(nCells,mCells,length(standardConductances));
        
        % Impose the same variability at every execution of the script
        load OriginalSeed
        % cellHeterSeed = rng;
        rng(cellHeterSeed);
        
        for i = nCells:-1:1
            for j = mCells:-1:1
                
                scale = exp(sigma(kk)*randn(1,length(standardConductances))); % compute the log-normal variability
                heterConductances = standardConductances .* scale; % scale the selected costants (permeabilities and conductances)
                %         heterConductances = round(heterConductances, 5 , 'significant');
                
                constStr.g_sus      = heterConductances(1);
                constStr.g_to       = heterConductances(2);
                constStr.g_CaL      = heterConductances(3) * pCal_scal;
                constStr.g_CaT      = heterConductances(4);
                constStr.g_Kr       = heterConductances(5);
                constStr.g_Ks       = heterConductances(6);
                constStr.g_b_Ca     = heterConductances(7);
                constStr.g_b_Na     = heterConductances(8);
                constStr.g_if       = heterConductances(9);
                constStr.g_st       = heterConductances(10);
                constStr.i_NaK_max  = heterConductances(11);
                constStr.kNaCa      = heterConductances(12);
                
                constVars(i,j) = constStr;
            end
        end
        % rng(cellHeterSeed);
        
        %     gCond = [constVars(:).g_sus, constVars(:).g_to, constVars(:).g_CaL, constVars(:).g_CaT, constVars(:).g_Kr, ...
        %         constVars(:).g_Ks, constVars(:).g_b_Ca, constVars(:).g_b_Na, constVars(:).g_if, constVars(:).g_st...
        %         constVars(:).i_NaK_max, constVars(:).kNaCa]';
        
        %% Solve the ODE
        % Preallocate vectors
        y = zeros(stateVectorDim,nCells,mCells,length(t),'gpuArray'); % state variable vector for 1 iteration
        dy = zeros(size(yOld),'gpuArray'); % variation vector
        stateVect = zeros(rep,stateVectorDim,nCells,mCells,round((length(t)-1)/underSamp)); % state variable vector for every iteraion
        
%         w = waitbar(0,'0%','Name','Solving ODEs');
disp('************Sim start**********')
        for idx = 1:(rep+ 100) % rep/underSamp)
            %%% Esplicit Euler fixed step
            y = Maltsev_GPU_2D_Euler_mex(t,fixedStep,y,yOld,dy,gapJunct,constVars, VmIndex); % update the states
            
            yOld = y(:,:,:,end); % the last state is the initial condition of the next iteration
            % Save the iteration in the stateVar vector
            stateVect(idx,:,:,:,:) = squeeze(gather(y(:,:,:,1:underSamp:end)));
            % Every gpuRes iteration,  save final conditions for
            % the next iterations, clear GPU memory and re-preallocate vectors
            if mod(idx, gpuRes) == 0 || idx == rep
                stateVect(idx,:,:,:,:) = gather(y(:,:,:,1:underSamp:end));
                reset(g)
                y = zeros(stateVectorDim,nCells,mCells,length(t),'gpuArray');
                dy = zeros(size(yOld),'gpuArray');
                yOld = squeeze(gpuArray(stateVect(idx,:,:,:,end))); %gpuarray
            end
            if mod(idx,100) == 0
                disp(['Simulating...',num2str(round(idx/(rep+100)*100)),'% ||| R: ',num2str(jj),'/',num2str(length(rGap)),...
                    ' ||| std: ',num2str(kk),'/',num2str(length(sigma))])
            end
%             waitbar(idx/rep,w,[sprintf('%3.0f',idx/(rep+rep/underSamp)*100),'%  - Sim: ', num2str(jj), '/', num2str(length(rGap))])
        end
%         delete(w)
        clear y, clear dy, clear constVars
        disp('----------Sim end----------')

        
        % % % % % Create the time vector and extract action potential vector from the state
        % % % % % variable vector; the for cycle is made to delete the initial sample of
        % % % % % every iteration since it conatins the final state of the precedent step,
        % % % % % so it would be a repetition. I assign the two AP vector samples the same
        % % % % % time so that they coincide.
        
        skip = (length(t)/underSamp) * gpuRes;
        stateVect = reshape(permute(squeeze(stateVect),[2,3,4,5,1]),stateVectorDim,nCells,mCells,[]);
        stateVect(:,:,:,skip+1:skip:end) = [];
        stateVect = stateVect(:,:,:,1:19999);
        
        hh = fixedStep*underSamp;
        tt = (0:hh:(rep*tSpan(2))-2*hh) /1e3;
                
        if flagFinalCond == 1 && rGap(jj) == inf
            finalCond = gather(yOld);
            finalCondName = ['initCondMaltsev-',num2str(sigma(kk)),'.mat'];
            save(finalCondName,'finalCond')
        end
        
        %% Save
        if pCal_scal ~= 1
            filename = ['Maltsev', num2str(nCells), 'x', num2str(mCells),'_PCal-',num2str(pCal_scal),'_',num2str(tSpan(2)*rep/1e3), 's', '_', num2str(rGap(jj)/1e9), 'Gohm_var-', num2str(sigma(kk)), '.mat'];            
        else
            filename = ['Maltsev', num2str(nCells), 'x', num2str(mCells), '_',num2str(tSpan(2)*rep/1e3), 's', '_', num2str(rGap(jj)/1e9), 'Gohm_', 'var-', num2str(sigma(kk)), '.mat'];
        end
        save(['Malt_Results/', filename],'tt','stateVect','-v7.3')
        
    end
end
clear
