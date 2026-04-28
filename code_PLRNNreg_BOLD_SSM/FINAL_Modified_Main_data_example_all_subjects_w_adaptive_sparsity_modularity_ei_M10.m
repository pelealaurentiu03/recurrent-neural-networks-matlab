%(c) Georgia Koppe, 2019
% Firstly modified by Andrei on 15.11.2025 – data-file selection + Schaefer atlas
% Modified by Andrei on 18.11.2025 - Implementing the sparsity prior (Batch Mode)
% Modified by Andrei on 27.11.2025 - Implementation of adaptive sparsity
% Modified by Laurentiu on 12.16.2025 - Modularity Prior
% Modified by Andrei on 18.12.2025 - Batch Processing Improvements + optimizations
% Modified by Lia on 21.12.2025 – Defining biologically plausible E/I ratio (Step 2.3)
% Modified by Lia on 24.12.2025 – Enforcing E/I sign constraints via projection on outgoing weights
% Modified by Lia on 04.01.2026 – Added verification and safety handling for E/I constraints during sparsity optimization
% -------------------------------------------------------------------------
clear; close all; clc
%-> directories 
thisFileDir = fileparts(mfilename('fullpath'));
cd(thisFileDir);
startpath = pwd;
patPLRNN = fullfile(startpath,'PLRNNreg_BOLD_SSM');
patLDS   = fullfile(startpath,'LDS_BOLD_SSM');
patData  = fullfile(startpath,'data');

if patData(end) ~= filesep
    patData = [patData filesep];
end
patOut   = fullfile(patData,'output');
patFigs  = fullfile(patOut,'figures');
patFigsMod = fullfile(patOut,'figures_modularity');
if ~isfolder(patOut),     mkdir(patOut);     end
if ~isfolder(patFigs),    mkdir(patFigs);    end
if ~isfolder(patFigsMod), mkdir(patFigsMod); end
addpath(patPLRNN);
addpath(patLDS);
addpath(patData);
fprintf('DEBUG patData = %s\n', patData);
fprintf('DEBUG #datafiles = %d\n', numel(dir(fullfile(patData,'datafile_*.mat'))));

%-> options
opt = 2; % 2 = use task inputs
M = 10;  % change to 20 to use the Atlas!

% ============================================================
% Step 2.3 (E/I): Define biologically plausible E/I ratio + assignment
% ============================================================
regEI = struct();
regEI.use_EI   = true;
regEI.EI_ratio = 0.80;                 % 80/20 split
nE = round(regEI.EI_ratio * M);
EI_sign = ones(M,1);
EI_sign(nE+1:end) = -1;               % last 20% inhibitory
regEI.EI_sign = EI_sign;              % +1 excitatory row, -1 inhibitory row
% -------------------------------------------------------------------------
% <<< DATA-FILE SELECTION >>>
% -------------------------------------------------------------------------
str = '*.mat';
files = dir([patData str]);
if isempty(files)
    files = dir(fullfile(patData,'datafile_*.mat'));
end
isDataFile = arrayfun(@(f) startsWith(f.name,'datafile_'), files);
files = files(isDataFile);
if isempty(files), error('No datafile_*.mat files found in %s', patData); end
nfiles = numel(files);
fprintf('Found %d subjects. Starting Batch Processing...\n', nfiles);
% ------------------------
% <<< SCHAEFER ATLAS >>>
% ------------------------
schaeferNiiFile = [patData 'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.nii'];
if ~exist(schaeferNiiFile,'file')
    error('Place Schaefer .nii in %s', patData);
end
info = niftiinfo(schaeferNiiFile);
parcelVolume = niftiread(info);
[nx,ny,nz] = size(parcelVolume);
transform = info.Transform.T;
inv_transform = inv(transform);
fprintf('Schaefer loaded: %dx%dx%d voxels (parcels 1-%d).\n', nx,ny,nz, max(parcelVolume(:)));

disp('--- NIfTI info for diagnosis ---');
disp('Affine transform (info.Transform.T):'); disp(info.Transform.T);
disp(['Qfactor: ' num2str(info.Qfactor)]);
disp(['SpaceUnits: ' info.SpaceUnits]);
disp(['TimeUnits: ' info.TimeUnits]);
disp('Raw pixdim:'); disp(info.raw.pixdim);
disp('--------------------------------');
% -------------------------------------------------------------------------
% ROI → voxel mapping (MNI→voxel)
% -------------------------------------------------------------------------
roiMNI = [-36,-4,52; 38,-2,50; ... 
         -8,32,20; 8,32,20; ...
        -48,38,20; 50,38,18; ...
        -42,30,32; 44,30,30; ...
        -50,22,10; 52,22,8; ...
        -44,22,-4; 46,22,-2; ...
        -30,52,0; 32,52,2; ...
        -28,-64,50; 30,-64,48; ...
        -50,-32,40; 52,-32,38; ...
         -4,-60,-20; 4,-60,-18];
roiNames = {'BA6_L','BA6_R','BA32_L','BA32_R','BA46_L','BA46_R','BA9_L','BA9_R',...
 'BA45_L','BA45_R','BA47_L','BA47_R','BA10_L','BA10_R','BA7_L','BA7_R',...
 'BA40_L','BA40_R','Cereb_L','Cereb_R'};
% --------------
% LUT FILE LOAD
% --------------
lutFile = fullfile(patData,'Schaefer2018_1000Parcels_7Networks_order.txt');
if ~exist(lutFile,'file')
    error('LUT file not found: %s', lutFile);
end
txt = fileread(lutFile);
fprintf('LUT loaded from: %s\n', lutFile);
parcelToNetwork = containers.Map('KeyType','double','ValueType','double');
if ~isempty(txt)
    lines = strsplit(txt,'\n');
    networkMap = containers.Map({'Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default'}, [1,2,3,4,5,6,7]);
    for i = 1:numel(lines)
        ln = strtrim(lines{i});
        if isempty(ln), continue; end
        parts = strsplit(ln);
        if numel(parts) < 5, continue; end
        pid = str2double(parts{1});
        label = parts{2};
        label_parts = strsplit(label,'_');
        if numel(label_parts) >= 3
            net_name = label_parts{3};
            if isKey(networkMap, net_name)
                net = networkMap(net_name);
                if ~isnan(pid) && ~isnan(net)
                    parcelToNetwork(pid) = net;
                end
            end
        end
    end
    fprintf('LUT parsed: %d mappings.\n', numel(keys(parcelToNetwork)));
else
    warning('No LUT text – using predefined networks.');
end

atlasName = 'Schaefer2018_1000Parcels_7Networks';
% -------------------------------------------------------------------------
% ROI→Module mapping: MNI→voxel→parcel→network (Pre-calculate best perm)
% -------------------------------------------------------------------------
mni2vox = @(coord) round( ...
    [ (coord(1) - info.Transform.T(4,1)) / info.Transform.T(1,1) + 1; ...
      (coord(2) - info.Transform.T(4,2)) / info.Transform.T(2,2) + 1; ...
      (coord(3) - info.Transform.T(4,3)) / info.Transform.T(3,3) + 1 ] );
rawVox = zeros(size(roiMNI,1),3);
for r = 1:size(roiMNI,1)
    vox = mni2vox(roiMNI(r,:)');
    vox(1) = max(1,min(vox(1),nx));
    vox(2) = max(1,min(vox(2),ny));
    vox(3) = max(1,min(vox(3),nz));
    rawVox(r,:) = vox';
end
permuts = [1 2 3; 1 3 2; 2 1 3; 2 3 1; 3 1 2; 3 2 1];
signs = [1 1 1; 1 1 -1; 1 -1 1; 1 -1 -1; -1 1 1; -1 1 -1; -1 -1 1; -1 -1 -1];
score = zeros(size(permuts,1), size(signs,1));
for pi = 1:size(permuts,1)
    for si = 1:size(signs,1)
        hits = 0;
        for r = 1:size(rawVox,1)
            vv = rawVox(r, permuts(pi,:)) .* signs(si,:);
            iv = max(1,min(round(vv(1)), nx));
            jv = max(1,min(round(vv(2)), ny));
            kv = max(1,min(round(vv(3)), nz));
            pid = double(parcelVolume(iv,jv,kv));
            if pid>0 && isKey(parcelToNetwork,pid)
                hits = hits + 1;
            end
        end
        score(pi,si) = hits;
    end
end
[maxhits, idx] = max(score(:));
[bestP, bestS] = ind2sub(size(score), idx);
bestPerm = permuts(bestP,:);
bestSign = signs(bestS,:);
fprintf('\n=== Optimal Schaefer voxel mapping found! ===\n');
fprintf(' Valid LUT matches = %d / %d ROIs\n', maxhits, size(rawVox,1));
% ==========================
% <<< MAIN BATCH LOOP >>>
% ==========================
for k = 1:nfiles
    
    try
        filename = files(k).name;
        
        rng(k, 'twister'); 
        fprintf('\n------------------------------------------------------------\n');
        fprintf('Loading subject %d/%d: %s (RNG Seed: %d)\n', k, nfiles, filename, k);
        fprintf('------------------------------------------------------------\n');
        
        %-> load of the fMRI data
        load(fullfile(patData, filename)) 
        X = PLRNN.data; R = PLRNN.rp; Inp= PLRNN.Inp;
        K = size(Inp,1); T = size(X,2); N = size(X,1); P = size(R,2); TR = PLRNN.preprocess.RT;
        %M=N;
        if opt==1, disp('caution: inputs set to 0!'); Inp = zeros(K,T); end
        
        % -------------------------------------------
        % <<< MASK CREATION BLOCK (Per Subject) >>>
        % -------------------------------------------
        moduleIdx = zeros(N,1);
        for i = 1:N
            vv = rawVox(i, bestPerm) .* bestSign;
            iv = max(1,min(round(vv(1)), nx));
            jv = max(1,min(round(vv(2)), ny));
            kv = max(1,min(round(vv(3)), nz));
            pid = double(parcelVolume(iv,jv,kv));
            if pid>0 && isKey(parcelToNetwork,pid)
                moduleIdx(i) = parcelToNetwork(pid);
            else
                moduleIdx(i) = 0;
            end
        end
        
        Q_mask = ones(M, M); 
        high_penalty = 10; 
        if M == N
            for i = 1:M
                for j = 1:M
                    if (moduleIdx(i) ~= moduleIdx(j)) || (moduleIdx(i) == 0) || (moduleIdx(j) == 0)
                        Q_mask(i,j) = high_penalty;
                    end
                end
            end
            fprintf('   [Modularity] Applied Atlas-based block structure to W.\n');
        else
            for i = 1:M
                for j = 1:M
                    if i ~= j, Q_mask(i,j) = high_penalty; end
                end
            end
            fprintf('   [Modularity] M!=N. Applied Diagonal-preference structure to W.\n');
        end
        
        % ------------------------
        % <<< TRAINING PLRNN >>>
        % ------------------------
        n = ceil(100*randn); 
        net = PLRNN_BOLD(M,N,T,TR,P,X,R,Inp,K);
        net.reg.modularity_mask = Q_mask;
        net.reg.use_modularity = true;
        net.init_pars(n);
        
        % --- OPTIMIZATIONS ---
        tol = 1e-2;    
        MaxIter_Init = 10; 
        eps = 1e-5; fixedS = 1; fixedB = 0; fixedG = 0; fixedC = (opt==1);
        CtrPar_Init = [tol MaxIter_Init eps fixedS fixedC fixedB fixedG];
        net.Sigma = eye(M);
        
        fprintf('   -> [Phase 1] Initialization (LDS + %d iters)... ', MaxIter_Init);
        addpath(patLDS);
        net = EMiter_LDS2pari1(net,CtrPar_Init,X,Inp,R);
        rmpath(patLDS);
        fprintf('Done.\n');
        
        % 1. Setup for modularity registry
        if fixedC, LMask = zeros(size([net.A net.W net.h])); else, LMask = zeros(size([net.A net.W net.h net.C])); end
        LMask(1:ceil(M/2),1:M)=-1; LMask(1:ceil(M/2),M+1:2*M)=1; LMask(1:ceil(M/2),2*M+1)=1;
        reg.Lreg = LMask; 
        reg.lambda = 100; 
        reg.tau = 100;
        reg.modularity_mask = Q_mask; 
        reg.use_modularity = true;
        % ============================================================
        % Step 2.3 (E/I): Attaching EI settings to reg (so EMiter / M-step can see it)
        % ============================================================
        reg.use_EI   = regEI.use_EI;
        reg.EI_ratio = regEI.EI_ratio;
        reg.EI_sign  = regEI.EI_sign;
        net.reg = reg; 
        net.Sigma = eye(M);
        
        % 2. Forcing stability which is important for M=20 E-Step
        net.A = 0.9 * eye(M); 
        net.W = 0.1 * randn(M,M); % Small random weights
        
        % 3. LOOSEN TOLERANCES
        tol_outer = 0.1;  
        tol_inner = 0.1;  
        MaxIter_Pass2 = 5; 
        flipAll = false;   
        CtrPar_Pass2 = [tol_outer MaxIter_Pass2 tol_inner eps flipAll fixedS fixedC fixedB fixedG];
        
        fprintf('   -> [Phase 2] Modularity Pre-training (%d iters, Loose Tolerance)... ', MaxIter_Pass2);
        net = EMiter3pari1(net, CtrPar_Pass2, X, Inp, R);
        fprintf('Done.\n');
       
        MaxIter_Anneal = 5; 
        fprintf('   -> [Phase 3] Sigma Annealing ');
        for sigmaVal = [0.1 0.01 0.001]
            net.Sigma = sigmaVal*eye(M);
            fixedB = 1; fixedG = 1;
        
            CtrPar_Anneal = [tol MaxIter_Anneal tol_inner eps flipAll fixedS fixedC fixedB fixedG];
            [net,LL,Ezi,Ephizi,Ephizij,Eziphizj,V] = EMiter3pari1(net, CtrPar_Anneal, X, Inp, R);
            fprintf('.'); 
        end
        
        % ---------------------------------------------
        % <<< 2. AUTOMATED ADAPTIVE SPARSITY SEARCH >>>
        % ---------------------------------------------
        target_density_min = 0.10; 
        target_density_max = 0.30;
        total_weights = M * M;
        target_nz_min = max(1, round(total_weights * target_density_min)); 
        target_nz_max = max(target_nz_min + 1, round(total_weights * target_density_max));
        
        max_rho_iter = 20; 
        rho_current = 0.01; 
        rho_step = 1.5; 
        rho_decay = 0.8; 
        weight_thresh = 1e-4;
        
        MaxIter_Search = 5; 
        tol_Search = 1e-2;
        
        CtrPar_Search = [tol_Search MaxIter_Search tol_Search eps false fixedS fixedC fixedB fixedG];
        
        fprintf('\n=== SPARSITY SEARCH ===\n');
        fprintf('Target: %d to %d non-zero weights.\n', target_nz_min, target_nz_max);
        
        net_sparse = net; 
        % ============================================================
        % Step 2.3 (E/I): Ensure net_sparse keeps EI metadata
        % ============================================================
        if isfield(net,'reg') && isfield(net.reg,'use_EI')
            net_sparse.reg.use_EI = net.reg.use_EI;
        end
        if isfield(net,'reg') && isfield(net.reg,'EI_sign')
            net_sparse.reg.EI_sign = net.reg.EI_sign;
        end
        net_sparse.reg.use_L1 = true;
        net_sparse.reg.Lreg = zeros(size([net.A net.W net.h net.C])); 
        net_sparse.reg.lambda = 0; 
        net_sparse.reg.tau = 10; 
        net_sparse.reg.modularity_mask = Q_mask; 
        net_sparse.reg.use_modularity = true;
        net_sparse.Sigma = 0.001 * eye(M);
        
        history_rho = []; history_nz = [];
        
        for iter = 1:max_rho_iter
            net_sparse.reg.rho = rho_current;
            
            [net_sparse, ~] = EMiter3pari1(net_sparse, CtrPar_Search, X, Inp, R);
            % ============================================================
            % Step 2.3 (E/I): Projection after each EM step (row-wise, off-diagonal)
            % ============================================================
            if isfield(net_sparse,'reg') && isfield(net_sparse.reg,'use_EI') && net_sparse.reg.use_EI ...
                    && isfield(net_sparse.reg,'EI_sign') && numel(net_sparse.reg.EI_sign) == M
            
                EI = net_sparse.reg.EI_sign(:);
                Wtmp = net_sparse.W;
            
                for ii = 1:M
                    idx = [1:ii-1, ii+1:M];
                    if EI(ii) > 0
                        Wtmp(ii,idx) = max(Wtmp(ii,idx), 0); % Excitatory rows >= 0
                    else
                        Wtmp(ii,idx) = min(Wtmp(ii,idx), 0); % Inhibitory rows <= 0
                    end
                end
            
                net_sparse.W = Wtmp;
            end
            W_curr = net_sparse.W;
            current_nz = nnz(abs(W_curr) > weight_thresh);
            sparsity_perc = 100 * current_nz / numel(W_curr);
            
            history_rho(end+1) = rho_current;
            history_nz(end+1)  = current_nz;
            
            fprintf('Iter %d: rho = %.4f | Non-zeros = %d (%.1f%%)\n', iter, rho_current, current_nz, sparsity_perc);
            
            if current_nz > target_nz_max
                rho_current = rho_current * rho_step;
            elseif current_nz < target_nz_min
                if current_nz == 0 && iter > 1 && history_nz(iter-1) == 0
                     rho_current = rho_current * 0.1; 
                else
                     rho_current = rho_current * rho_decay;
                end
            else

                fprintf('\n>>> SUCCESS: Found optimal rho = %.4f. Polishing model (Debiasing)...\n', rho_current);
                
               
                W_mask = (abs(net_sparse.W) > weight_thresh);
                
             
                net_sparse.reg.rho = 0; 
                net_sparse.reg.use_L1 = false; 
                
              
                net_sparse.reg.modularity_mask = zeros(M,M); 
                net_sparse.reg.modularity_mask(~W_mask) = 1e9;
                net_sparse.reg.use_modularity = true; 
                
                CtrPar_Polish = [1e-3 10 1e-3 eps false fixedS fixedC fixedB fixedG];
                [net_sparse, ~] = EMiter3pari1(net_sparse, CtrPar_Polish, X, Inp, R);
                
                break;
            end
        end
        
        % -------------------------------------------------------------------------
        % <<< RESULTS VISUALIZATION & SAVE  >>>
        % -------------------------------------------------------------------------
        W_final = net_sparse.W;
        W_final(abs(W_final) < weight_thresh) = 0; 
        % ============================================================
        % Step 2.3 (E/I): FINAL projection on W_final (off-diagonal only)
        % ============================================================
        if isfield(net_sparse,'reg') && isfield(net_sparse.reg,'use_EI') && net_sparse.reg.use_EI ...
                && isfield(net_sparse.reg,'EI_sign') && numel(net_sparse.reg.EI_sign) == M
        
            EI = net_sparse.reg.EI_sign(:);
            for ii = 1:M
                idx = [1:ii-1, ii+1:M]; 
                if EI(ii) > 0
                    W_final(ii,idx) = max(W_final(ii,idx), 0);
                else
                    W_final(ii,idx) = min(W_final(ii,idx), 0);
                end
            end
        end
        net_sparse.W = W_final;
        % ============================================================
        % Step 2.3 (E/I): Quick verification (off-diagonal)
        % ============================================================
        if isfield(net_sparse,'reg') && isfield(net_sparse.reg,'EI_sign')
            Wcheck = net_sparse.W;
            Wcheck(eye(M)==1) = 0;
            E = find(net_sparse.reg.EI_sign > 0);
            I = find(net_sparse.reg.EI_sign < 0);
        
            fprintf('   [E/I CHECK] Min(E rows)=%g | Max(I rows)=%g\n', ...
                min(Wcheck(E,:),[],'all'), max(Wcheck(I,:),[],'all'));
        end
        
        disp('Final Sparse W matrix saved.');

        hFig = figure('Visible','off','Position',[100 100 800 400],'Color','w');
        subplot(1,2,1);
        
        max_val = max(abs(W_final(:)));
        if max_val == 0, max_val = 1; end 
        imagesc(W_final, [-1 1]*max_val); 
        
        colormap(parula); colorbar; axis square equal tight;
        title(sprintf('Subj %d Final W (M=%d, \\rho=%.3f)\n%d Non-Zeros', k, M, net_sparse.reg.rho, nnz(W_final)));
        subplot(1,2,2);
        yyaxis left; plot(history_rho, '-o'); ylabel('Rho');
        yyaxis right; plot(history_nz, '-s'); ylabel('NZ'); yline(target_nz_max,'--g'); yline(target_nz_min,'--g');
        title('Optimization Trajectory');
        
        % Save Figure
        figName = fullfile(patFigsMod, sprintf('Subject_%03d_Sparsity_Modularity_EI_M10.png', k));
        saveas(hFig, figName);
        close(hFig);
        
        % Save Data
        fileOut = [patOut 'outputfile' num2str(k) '_w_sparsity_modularity_ei_M10.mat'];
        save(fileOut,'net','net_sparse','LL','Ezi','Ephizi','Ephizij','Eziphizj','V',...
            'moduleIdx','atlasName','roiMNI');
        disp(['Results saved: ' fileOut]);
    
    catch ME
        fprintf('\n!!! ERROR processing Subject %d: %s\n', k, ME.message);
        disp('Continuing to next subject...');
    end
end 
fprintf('\n=== ALL SUBJECTS PROCESSED SUCCESSFULLY ===\n');