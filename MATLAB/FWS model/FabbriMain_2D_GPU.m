% close all
clear
clc

% Reset the GPU
g = gpuDevice(1);

% Set C++ compiler path
setenv('BASH_ENV','~/.bash_profile');
p = getenv('PATH');
setenv('PATH', [p ':/usr/local/cuda-10.1/bin'])

%% Simulation parameters
nCells = 50;
mCells = 50;

% Gap junction
Cm = 57*1e-12; % pF
rGap = 1000 * 1e6; %[10, 100, 1000, 10000, inf]*1e6; % MOhm
sigma = 0.2;  % [0, 0.05, 0.1, 0.1873, 0.3, 0.4]; % n% variability (standard deviation)
flagFinalCond = 0;
pCal_scal = 1;% [1, 0.9, 0.75, 0.5]; % 0.5; %[1, 0.8, 0.5];

tSpan = [0 5]; % simulate 5 s every iteration
rep = 1; % do n iteration: rep*tSpan = total simulation time
fixedStep = 1e-5; % integration step
t = 0:fixedStep:tSpan(2)-fixedStep;
underSamp = 100; % keep 1 sample every 100 (about 1 ms of sampling step having fixedStep 1.25e-5 s)

fromMDP = '';
simType = 'WT';
VmIndex = 15; % Membrane potential is the 15th variable of the state vector

%%

for ww = 1:length(rGap)
    for kk = 1:length(sigma)
        
        %% Initial conditions
%         load initCond.mat
        stateVectorDim = 33; % size(y_start,2);
        %         % Sample the y_start data file randomly to the the initial conditions of the cells
        %         initCondSeed = RandStream('mlfg6331_64');
        %         startPos = datasample(initCondSeed,1:length(y_start),nCells*mCells); % s, always keeps the same random sample
        %         %%% SAME INIT CONDS
        %         % startPos(2:end) = startPos(1);
        %         % numSim = 3;
        %         %%%
        %         % yOld = y_start(startPos,:,:)';
        %         yOld = reshape(y_start(startPos,:,:)',stateVectorDim,nCells,mCells);
        %         % Transfer data to the GPU
        
        if flagFinalCond == 1 %% && rGap(ww) == inf                    
            load InitCondEug %% initCondFabbri100.4-0.1.mat%% 
            yOld = finalCond;
        elseif strcmpi(fromMDP, 'MDP')
            sigma_lab = num2str(sigma);
            initCondFile = ['initPos_lastMDP_Rinf_s0',num2str(sigma_lab(3)),'.mat'];
            load(initCondFile)
            yOld = initCondVar;
            yOld = permute(yOld(1:mCells, 1:nCells,:), [3, 1, 2]);
        else
            initCondFile = ['initCondFabbri-',num2str(sigma(kk)),'.mat'];
            load(initCondFile)
            yOld = finalCond;
            yOld = yOld(:,1:mCells, 1:nCells);
        end
        %         yOld = round(yOld,5,'significant');
        
        yOld = gpuArray(yOld);
        
        %% Variability
        modelConstants % Call the struct containing the costants of the model
        
        standardConductances = [constStr.P_CaL, constStr.P_CaT, constStr.g_KACh, constStr.g_Kr, constStr.g_Ks_, ...
            constStr.g_Kur, constStr.g_Na, constStr.g_Na_L, constStr.g_f, constStr.g_to...
            constStr.i_NaK_max, constStr.K_NaCa];
        heterConductances = zeros(nCells,mCells,length(standardConductances));
        
        % % Impose the same variability at every execution of the script
        %         cellHeterSeed = rng;
        load('OriginalSeed.mat');
        rng(cellHeterSeed);
        
        gapJunct = ones(nCells,mCells)*Cm*rGap(ww);
        
        for i = nCells:-1:1
            for j = mCells:-1:1
                
                scale = exp(sigma(kk)*randn(1,length(standardConductances))); % compute the log-normal variability
                heterConductances = standardConductances .* scale; % scale the selected costants (permeabilities and conductances)
%                 heterConductances = round(standardConductances .* scale,5,'significant'); % scale the selected costants (permeabilities and conductances)

                constStr.P_CaL      = heterConductances(1) * pCal_scal;
                constStr.P_CaT      = heterConductances(2);
                constStr.g_KACh     = heterConductances(3);
                constStr.g_Kr       = heterConductances(4);
                constStr.g_Ks_      = heterConductances(5);
                constStr.g_Kur      = heterConductances(6);
                constStr.g_Na       = heterConductances(7);
                constStr.g_Na_L     = heterConductances(8);
                constStr.g_f        = heterConductances(9);
                constStr.g_to       = heterConductances(10);
                constStr.i_NaK_max  = heterConductances(11);
                constStr.K_NaCa     = heterConductances(12);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%% MUTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if strcmpi(simType, 'DM')
                    constStr(:).g_f = constStr(:).g_f * 0.08; % double mutant
                elseif strcmpi(simType, 'HM45')
                    constStr(:).g_f = constStr(:).g_f * 0.45; % double mutant
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                constVars(i,j) = constStr;
            end
        end
        % rng(cellHeterSeed);
        %         disp(cell2mat({constVars(:).g_f}))
        %         a = cell2mat({constVars(:).P_CaL});
        % constVars = constStr;
%         
%          gs = [constVars(:).P_CaL; constVars(:).P_CaT; constVars(:).g_KACh; ...
%              constVars(:).g_Kr; constVars(:).g_Ks_; constVars.g_Kur; constVars(:).g_Na;...
%              constVars(:).g_f; constVars(:).g_to; constVars(:).i_NaK_max; constVars(:).K_NaCa];
%         
        
        %% Solve the ODE
        % Preallocate vectors
        y         = zeros(stateVectorDim,nCells,mCells,round((length(t))/underSamp)); % state variable vector for 1 iteration
        dy        = zeros(size(yOld)); % variation vector
        stateVect = zeros(stateVectorDim,nCells,mCells,round(length(t)/underSamp),rep); % state variable vector for every iteraion
        
        y  = gpuArray(y);
        dy = gpuArray(dy);
        
        %         w = waitbar(0,'0%','Name','Solving ODEs');
        disp('************Sim start**********')
        for idx = 1:rep %(rep/underSamp)
            %%% Esplicit Euler fixed step
            y = GPU_2D_Euler_mex(t,fixedStep,y,yOld,dy,gapJunct,constVars, underSamp); % update the states
         
            yOld = y(:,:,:,end); % the last state is the initial condition of the next iteration
            % Save the iteration in the stateVar vector
            stateVect(:,:,:,:,idx) = squeeze(gather(y));
            % Every gpuRes iteration,  save final conditions for
            % the next iterations, clear GPU memory and re-preallocate vectors
            
            reset(g)
            y    = zeros(stateVectorDim,nCells,mCells,round((length(t))/underSamp)); % state variable vector for 1 iteration
            dy   = zeros(size(yOld)); % variation vector
            yOld = squeeze(gpuArray(stateVect(:,:,:,end,idx)));
            
            y    = gpuArray(y);
            dy   = gpuArray(dy);
            yOld = gpuArray(yOld);
            disp(['Simulating...',num2str(round(idx/rep*100)),'% ||| R: ',num2str(ww),'/',num2str(length(rGap)),...
                ' ||| std: ',num2str(kk),'/',num2str(length(sigma))])

        end
        %         delete(w)
        clear y, clear dy, clear constVars
        disp('----------Sim end----------')
        
        hh = fixedStep*underSamp;
        tt = 0:hh:(rep*tSpan(2))- hh;
        AP = reshape(squeeze(stateVect(VmIndex,:,:,:,:)), mCells,nCells, []);
        
        figure
        plot(tt, squeeze(AP(:,25,:)))
        
        if flagFinalCond == 1 % && rGap(ww) == inf
            finalCond = gather(yOld);
            finalCondName = ['initCondFabbri100.5-',num2str(sigma(kk)),'.mat'];
            save(finalCondName,'finalCond')
        end
        
        %% Save
%         if sigma(kk) == 0
%             filename = [fromMDP,'Fabbri', num2str(nCells), 'x', num2str(mCells),'_',num2str(tSpan(2)*rep), 's', '_', num2str(rGap(ww)/1e6), 'Mohm_OMO_(', simType, ').mat'];
%         elseif pCal_scal ~= 1
%             filename = [fromMDP,'Fabbri', num2str(nCells), 'x', num2str(mCells),'_PCal-',num2str(pCal_scal),'_',num2str(tSpan(2)*rep), 's', '_', num2str(rGap(ww)/1e6), 'Mohm_var-', num2str(sigma(kk)), '.mat'];
%         else
%             filename = [fromMDP,'Fabbri', num2str(nCells), 'x', num2str(mCells),'_',num2str(tSpan(2)*rep), 's', '_', num2str(rGap(ww)/1e6), 'Mohm_var-', num2str(sigma(kk)), '.mat'];
%         end
%         save(['/home/eugenior/Desktop/SAN_project/Fabbri_Results/', filename],'tt','AP')
% %         save(['/home/eugenior/Desktop/SAN_project/Fabbri_Results/', filename],'AP')
%                 clear time
    end
end

% clear
