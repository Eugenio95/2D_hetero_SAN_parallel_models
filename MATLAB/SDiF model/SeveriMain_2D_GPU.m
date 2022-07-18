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

tSpan = [0 0.5e-2]; % simulate 10 ms every iteration
rep = 4000; % do n iteration: rep*tSpan = total simulation time

fixedStep = 0.5e-5; % integration step
t = 0:fixedStep:tSpan(2)-fixedStep;
underSamp = 2*100; % keep 1 sample every 100 (about 1 ms of sampling step having fixedStep 1.25e-5 s)

VmIndex = 17; % Membrane potential is the 15th variable of state vector

gpuRes = 10; % Reset the GPU (clear its memory every 10 iteration, otherwise out of stak error)

% Gap junction
Cm = 32*1e-12; % pF
rGap = inf; %[10, 100, 1000, 10000, inf]*1e6; % MOhm % MANCA INF!!
% sigma = [0.3, 0.5]; % n% variability (standard deviation)
sigma = ['5']; %['1','2','3','4','5'];
flagFinalCond = 0;
pCal_scal = 1; % [1, 0.8, 0.5]


for kk = 1:length(sigma)
    for jj = 1:length(rGap)
        
        gapJunct = ones(nCells,mCells)*Cm*rGap(jj);
        
        %% Initial conditions
%         load Sev_initSamp.mat
        stateVectorDim = 33; %size(Sev_y_start,2);
        % % % % % % Sample the y_start data file randomly to the the initial conditions of the cells
        % % % % % initCondSeed = RandStream('mlfg6331_64');
        % % % % % startPos = datasample(initCondSeed,1:length(Sev_y_start),nCells*mCells); % s, always keeps the same random sample
        % % % % % %%% SAME INIT CONDS
        % % % % % % startPos(2:end) = startPos(1);
        % % % % % % numSim = 3;
        % % % % % %%%
        % % % % % % yOld = y_start(startPos,:,:)';
        % % % % % yOld = reshape(Sev_y_start(startPos,:,:)',stateVectorDim,nCells,mCells);
        
        if flagFinalCond == 1 && rGap(jj) == inf
        load Sev_Init_Cond
        yOld = yStart;
        else
            initCondFile = ['initCondSeveri-',num2str(sigma(kk)),'.mat'];
            load(initCondFile)
            yOld = finalCond;
        end
        % yOld = round(Malt_InitCond, 5, 'significant');
        
        % Transfer data to the GPU
        yOld = gpuArray(yOld);
        
        %% Variability
        
        %{
    SeveriConstants % Call the struct containing the costants of the model
    
    standardConductances = [constStr.P_CaL; constStr.P_CaT; constStr.g_Kr; ...
        constStr.g_Ks; constStr.g_to; constStr.g_f; constStr.i_NaK_max; ...
        constStr.K_NaCa; constStr.g_KACh; constStr.g_Na];
    
    % Impose the same variability at every execution of the script
    load OriginalSeed
    % cellHeterSeed = rng;
    rng(cellHeterSeed);
    
    for i = nCells:-1:1
        for j = mCells:-1:1
            
            scale = exp(sigma(kk)*randn(1,length(standardConductances))); % compute the log-normal variability
            heterConductances = standardConductances .* scale; % scale the selected costants (permeabilities and conductances)
            %         heterConductances = round(heterConductances, 5 , 'significant');
            
            constStr.P_CaL      = heterConductances(1) * pCal_scal;
            constStr.P_CaT      = heterConductances(2);
            constStr.g_Kr       = heterConductances(3);
            constStr.g_Ks       = heterConductances(4);
            constStr.g_to       = heterConductances(5);
            constStr.g_f        = heterConductances(6);
            constStr.i_NaK_max  = heterConductances(7);
            constStr.K_NaCa     = heterConductances(8);
            constStr.g_KACh     = heterConductances(9);
            constStr.g_Na       = heterConductances(10);
            
            constVars(i,j) = constStr;
        end
    end
    % rng(cellHeterSeed);
    
    gs = [g_CaL; g_CaT; g_Kr; g_Ks; g_to; g_if; i_NaK_max; K_NaCa; g_KACh; g_Na];

        g = [constVars(:).P_CaL, constVars(:).P_CaT, constVars(:).g_KACh, constVars(:).g_Kr,...
            constVars(:).K_NaCa, constVars(:).i_NaK_max, constVars(:).g_Na, constVars(:).g_to]';
    
        %}
        %% Conduttanze Chiara
        
        %%{
        SeveriConstants
        
        fileConduct = ['gCs_Sev_0', sigma(kk), '.mat'];
        load(fileConduct)
        
        for i = nCells:-1:1
            for j = mCells:-1:1
                
                constStr.P_CaL     = cond(1,i,j) * pCal_scal;
                constStr.P_CaT     = cond(2,i,j);
                constStr.g_Kr      = cond(3,i,j);
                constStr.g_Ks      = cond(4,i,j);
                constStr.g_to      = cond(5,i,j);
                constStr.g_f      = cond(6,i,j);
                constStr.i_NaK_max = cond(7,i,j);
                constStr.K_NaCa    = cond(8,i,j);
                constStr.g_KACh    = cond(9,i,j);
                constStr.g_Na      = cond(10,i,j);
                
                constVars(i,j) = constStr;
                
            end
        end
        
        gs = [constVars(:).P_CaL; constVars(:).P_CaT; constVars(:).g_Kr; constVars(:).g_Ks; ...
            constVars(:).g_to; constVars(:).g_f; constVars(:).i_NaK_max; constVars(:).K_NaCa; ...
            constVars(:).g_KACh; constVars(:).g_Na];
        
        %%}
        
        %% Solve the ODE
        % Preallocate vectors
        y = zeros(stateVectorDim,nCells,mCells,length(t),'gpuArray'); % state variable vector for 1 iteration
        dy = zeros(size(yOld),'gpuArray'); % variation vector
        stateVect = zeros(rep,stateVectorDim,nCells,mCells,round((length(t)-1)/underSamp)); % state variable vector for every iteraion
        
%         w = waitbar(0,'0%','Name','Solving ODEs');
        disp('************Sim start**********')
        for idx = 1:(rep+100)
            %%% Esplicit Euler fixed step
            y = Severi_GPU_2D_Euler_mex(t,fixedStep,y,yOld,dy,gapJunct,constVars, VmIndex); % update the states
            
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
                yOld = gpuArray(squeeze((stateVect(idx,:,:,:,end)))); % gpuarray
            end
            if mod(idx,100) == 0
                disp(['Simulating...',num2str(round(idx/(rep+100)*100)),'% ||| R: ',num2str(jj),'/',num2str(length(rGap)),...
                    ' ||| std: ',num2str(kk),'/',num2str(length(sigma))])
            end
%             waitbar(idx/rep,w,[sprintf('%3.0f',idx/(rep+rep/100)*100),'% - Sim: ', num2str(jj), '/', num2str(length(rGap))])
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
        tt = 0:hh:(rep*tSpan(2))-2*hh;
        
        if flagFinalCond == 1 && rGap(jj) == inf
            finalCond = gather(yOld);
            finalCondName = ['initCondSeveri-',num2str(sigma(kk)),'.mat'];
            save(finalCondName,'finalCond')
        end
        
        %% Save
        if pCal_scal ~= 1
            filename = ['Severi', num2str(nCells), 'x', num2str(mCells),'_PCal-',num2str(pCal_scal),'_',num2str(tSpan(2)*rep), 's', '_', num2str(rGap(jj)/1e6), 'Mohm_var-0.', num2str(sigma(kk)), '.mat'];
        else
            filename = ['Severi', num2str(nCells), 'x', num2str(mCells), '_',num2str(tSpan(2)*rep), 's', '_', num2str(rGap(jj)/1e6), 'Mohm_', 'var-0.', num2str(sigma(kk)), '.mat'];
        end
        save(['Sev_Results/', filename],'tt','stateVect','-v7.3')
                
    end
end

clear
