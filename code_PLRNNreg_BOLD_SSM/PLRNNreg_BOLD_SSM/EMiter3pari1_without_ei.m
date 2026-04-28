% =========================================================================
% FINAL MODIFIED FILE: EMiter3pari1.m
% Modified by Andrei on 18.11.2025 - Modified the original code by adding
%    the M- step selection so that it won't be overwritten by the orignal
%    baseline code that implements the regularization.
% Modified by Laurentiu on 16.12.2025 Modularity Implementation
% =========================================================================
% Optimized by Andrei on 18.12.2025:
% 1. Routes to Optimized M-Step for Speed.
% 2. Includes Debug Prints.
% 3. AUTO-FIXES LMask Dimensions to prevent "incompatible sizes" crash.
% 4. AUTO-FIXES Input Types (Unwraps Cells) for ComplLogLike.
% =========================================================================
function [net,LL,Ezi,Ephizi,Ephizij,Eziphizj,Vest]=EMiter3pari1(net,CtrPar,X,Inp,rp)
i=1; LLR=1e8; LL=[]; 
Ezi=net.Z;
tol=CtrPar(1); MaxIter=CtrPar(2); tol2=CtrPar(3); eps=CtrPar(4);
flipAll=CtrPar(5); fixedS=CtrPar(6); fixedC=CtrPar(7); fixedB=CtrPar(8); fixedG=CtrPar(9);   

while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<MaxIter
    
    % --- DEBUG: E-Step Start ---
    fprintf('   [Iter %d] Starting E-Step (Kalman)...', i);
    
    % E-step
    [Ezi,U,~,~]=StateEstimPLRNN2(net,Inp,X,rp,Ezi,[],tol2,eps,flipAll);
    [Ephizi,Ephizij,Eziphizj,Vest]=ExpValPLRNN3(net,Ezi,U);
    
    % --- DEBUG: E-Step End ---
    fprintf(' Done. Starting M-Step...');
    
    % =====================================================================
    % TEAM 2 CONTRIBUTION: SELECT M-STEP (Baseline vs Sparse)
    % =====================================================================
    % M-step Logic (Route to Optimized Function if needed)
    use_our_method = false;
    % OUR NEW L1-SPARSE M-STEP (main project contribution)
    if isfield(net.reg, 'use_L1') && net.reg.use_L1, use_our_method = true; fprintf(' [Our L1-Sparse M-STEP Implementation: Active] '); end
    % OUR NEW MODULARITY IMPLEMENTATION (main project contribution)
    if isfield(net.reg, 'use_modularity') && net.reg.use_modularity, use_our_method = true; fprintf(' [Our Modularity Implementation: Active] '); end
    
    if use_our_method
        % USAGE OF OUR IMPLEMENTATIONS
        net = Our_ParamEstimPLRNN_sp(net, Ezi, Vest, Ephizi, Ephizij, Eziphizj, X, Inp, rp, ...
                                     fixedS, fixedC, fixedG, fixedB);
    else
    % ORIGINAL KOPPE 2019 M-STEP (exact reproduction)
        net = ParamEstimPLRNN_sp(net, Ezi, Vest, Ephizi, Ephizij, Eziphizj, X, Inp, rp, ...
                                 fixedS, fixedC, fixedG, fixedB);
        fprintf(' [Using original M-STEP implementation.] ');
    end
    
    % --- DEBUG: M-Step End ---
    fprintf(' Done.');
    
    % =====================================================================
    % CRASH PREVENTER 1: Auto-Resize LMask to match Parameters
    % This fixes the "Arrays have incompatible sizes" error.
    % =====================================================================
    if isfield(net, 'reg') && isfield(net.reg, 'Lreg')
        % 1. Reconstruct current parameter matrix 'L' shape
        if fixedC
            L_curr = [net.A net.W net.h];
        else
            L_curr = [net.A net.W net.h net.C];
        end
        
        mask_size = size(net.reg.Lreg);
        param_size = size(L_curr);
        
        % 2. If sizes differ, resize the mask dynamically
        if ~isequal(mask_size, param_size)
            new_mask = zeros(param_size);
            
            % Copy strictly overlapping region to preserve regularization logic
            rows = min(mask_size(1), param_size(1));
            cols = min(mask_size(2), param_size(2));
            new_mask(1:rows, 1:cols) = net.reg.Lreg(1:rows, 1:cols);
            
            % Update the net object
            net.reg.Lreg = new_mask;
        end
    end
    
    % =====================================================================
    % CRASH PREVENTER 2: Unwrap Cell Arrays for ComplLogLike
    % This fixes the "Incorrect dimensions for matrix multiplication" error.
    % =====================================================================
    Inp_mat = Inp;
    X_mat = X;
    if iscell(Inp), Inp_mat = Inp{1}; end
    if iscell(X), X_mat = X{1}; end

    % Compute Log-Likelihood (Now Safe)
    LL(i)=net.ComplLogLike(Inp_mat, X_mat, Ezi, rp, fixedC);   
    
    if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end 
    fprintf('\n'); 
    i=i+1;
end

if isempty(LL)
    disp('LL calculation failed for all iterations.');
else
    disp(['LL, # iterations = ' num2str(LL(end)) ', ' num2str(i-1)]);
end

% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
% adapted 2019 Georgia Koppe, Dept. Theoretical Neuroscience, Central 
% Institute of Mental Health, Heidelberg University