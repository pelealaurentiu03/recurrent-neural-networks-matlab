% =========================================================================
% Modified by Andrei on 18.11.2025 - Modified the original code by adding
%    the M- step selection so that it won't be overwritten by the orignal
%    baseline code that implements the regularization.
% =========================================================================
function [net,LL,Ezi,Ephizi,Ephizij,Eziphizj,Vest]=EMiter3pari1(net,CtrPar,X,Inp,rp)
i=1; LLR=1e8; LL=[]; 
Ezi=net.Z;
tol=CtrPar(1);
MaxIter=CtrPar(2);
tol2=CtrPar(3);
eps=CtrPar(4);
flipAll=CtrPar(5);  % flipOnit option not yet included; set flipAll to false !
fixedS=CtrPar(6);   % S to be considered fixed or to be estimated
fixedC=CtrPar(7);   % C to be considered fixed or to be estimated
fixedB=CtrPar(8);   % B to be considered fixed or to be estimated
fixedG=CtrPar(9);   % G to be considered fixed or to be estimated

while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<MaxIter
    %E-step----------------------------------------------------------------
    [Ezi,U,~,~]=StateEstimPLRNN2(net,Inp,X,rp,Ezi,[],tol2,eps,flipAll);
    [Ephizi,Ephizij,Eziphizj,Vest]=ExpValPLRNN3(net,Ezi,U);
    
    %M-step----------------------------------------------------------------
    % =====================================================================
    % TEAM 2 CONTRIBUTION: SELECT M-STEP (Baseline vs Sparse)
    % =====================================================================
    if isfield(net.reg, 'use_L1') && net.reg.use_L1 == true
        % OUR NEW L1-SPARSE M-STEP (main project contribution)
        net = Our_ParamEstimPLRNN_sp(net, Ezi, Vest, Ephizi, Ephizij, Eziphizj, X, Inp, rp, ...
                                     fixedS, fixedC, fixedG, fixedB);
        fprintf('   → Using OUR L1-sparse M-step (rho = %.2f)\n', net.reg.rho);
    else
        % ORIGINAL KOPPE 2019 M-STEP (exact reproduction)
        net = ParamEstimPLRNN_sp(net, Ezi, Vest, Ephizi, Ephizij, Eziphizj, X, Inp, rp, ...
                                 fixedS, fixedC, fixedG, fixedB);
    end
    % =====================================================================

    %compute log-likelihood
    LL(i)=net.ComplLogLike(Inp,X,Ezi,rp,fixedC);   
   if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end 
   i=i+1;
end
disp(['LL, # iterations = ' num2str(LL(end)) ', ' num2str(i-1)]);

% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
% adapted 2019 Georgia Koppe, Dept. Theoretical Neuroscience, Central 
% Institute of Mental Health, Heidelberg University