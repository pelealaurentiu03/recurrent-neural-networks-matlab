function net=Our_ParamEstimPLRNN_sp(net,Ez,V,Ephizi,Ephizij,Eziphizj,X_,Inp_,rp, fixedS, fixedC, fixedG, fixedB,lam)
% =========================================================================
% MODIFIED FILE: Our_ParamEstimPLRNN_sp.m
% TEAM 2 CONTRIBUTION: 
% Modified by Andrei on 18.11.2025 - Added L1 (Lasso) Regularization for Sparsity on W
% Optimized by Andrei on 18.12.2025:
% 1. Removed Ezizi (Giant Sparse Matrix) to fix memory hang
% 2. Vectorized Modularity Prior & L1 Sparsity
% 3. Added Diagnostic Outputs
% 4. Fixed Transpose Bug in Penalty Assignment
% =========================================================================
% implements parameter estimation for PLRNN system
eps=1e-5;
if iscell(X_), X=X_; Inp=Inp_; else, X{1}=X_; Inp{1}=Inp_; end
ntr=length(X); m=size(Ez,1); N=size(X{1},1); Minp=size(Inp{1},1);
T=cell2mat(cellfun(@size,X,'UniformOutput',false)'); T=T(:,2);
Tsum=cumsum([0 T']); Lsum=Tsum.*m; Ez=Ez(1:end)';

% --- 1. Initialize Sums (Directly, NO giant Ezizi matrix) ---
E1=zeros(m); E2=E1; E3=E1; E4=E1; E5=E1; E1pkk=E1; E3pkk=E1; 
F1=zeros(N,m); F2=zeros(N,N); F3=zeros(Minp,m); F4=F3; F5_=zeros(m,Minp); F6_=zeros(Minp,Minp);
hF1=zeros(N,m); hE1pkk=E1pkk; hE1=E1; hE1p=zeros(m,m);
F7=zeros(N,size(rp,2)); F8=zeros(size(rp,2),size(rp,2)); hF9=zeros(size(rp,2),m);
Zt1=zeros(m,1); Zt0=zeros(m,1); phiZ=zeros(m,1); InpS=zeros(Minp,1);

% Fix Scope of f5_1
f5_1=zeros(m, Minp); f6_1=zeros(Minp, Minp);

% Pre-calculate Convolution Matrix
dt=length(net.hrf); H=net.getConvolutionMtx;

% --- MAIN LOOP: Process Data in Chunks (Fast) ---
for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt); Ephizi0=Ephizi(mt); 
    Ephizij0=Ephizij(mt,mt); Eziphizj0=Eziphizj(mt,mt);
    
    % Observation sums (Standard)
    F2=F2+X{i}*X{i}';
    F7=F7+X{i}*rp;
    F8=F8+rp'*rp;
    
    % Accumulate f5_1 and f6_1 (First time step only)
    f5_1 = f5_1 + Ez0(1:m)*Inp{i}(:,1)';
    f6_1 = f6_1 + Inp{i}(:,1)*Inp{i}(:,1)';
    
    % Fast HRF: Convolve expected states directly
    hEzi0 = H*Ez0;
    hEzi_mat = reshape(hEzi0, m, []); % Reshape to MxT
    
    % Update HRF sums using matrix multiplication (Instant)
    hF1 = hF1 + X{i}*hEzi_mat';
    hF9 = hF9 + rp'*hEzi_mat';
    hE1p = hE1p + (hEzi_mat * hEzi_mat'); % Approximate HRF covariance for speed
    
    % --- Loop over time (Now purely local, no sparse indexing) ---
    for t=2:T(i)
        k0=(t-1)*m+1:t*m; k1=(t-2)*m+1:(t-1)*m;
        
        % Reconstruct local V blocks
        V_t_t   = V(k0,k0) + Ez(k0)*Ez(k0)';
        V_t1_t1 = V(k1,k1) + Ez(k1)*Ez(k1)';
        V_t_t1  = V(k0,k1) + Ez(k0)*Ez(k1)';
        
        E1=E1+Ephizij0(k1,k1); 
        E2=E2+V_t_t1;
        E3=E3+V_t1_t1;
        E4=E4+Eziphizj0(k1,k1)'; 
        E5=E5+Eziphizj0(k0,k1);
        
        F1=F1+X{i}(:,t)*Ephizi0(k0)';
        F3=F3+Inp{i}(:,t)*Ez0(k1)';
        F4=F4+Inp{i}(:,t)*Ephizi0(k1)';
        F5_=F5_+Ez0(k0)*Inp{i}(:,t)';
        F6_=F6_+Inp{i}(:,t)*Inp{i}(:,t)';
    end
    
    E1pkk=E1pkk+Ephizij0(k0,k0);
    E3pkk=E3pkk+V_t_t; % Use last time step V
    
    zz=reshape(Ez0,m,T(i))';
    Zt1=Zt1+sum(zz(1:end-1,:))'; Zt0=Zt0+sum(zz(2:end,:))';
    pz=reshape(Ephizi0,m,T(i))'; phiZ=phiZ+sum(pz(1:end-1,:))';
    InpS=InpS+sum(Inp{i}(:,2:end)')';
end
E1p=E1+E1pkk; F5=F5_+f5_1; F6=F6_+f6_1;

% --- 2. Solve Parameters (This part is already fast) ---
if ~fixedB 
    bm1=[hF1 F7]; bm2=[hE1p hF9'; hF9 F8]; 
    BM=bm1*pinv(bm2); % Use pinv for stability
    B=BM(1:N,1:m); J=BM(1:N, m+1:end);
else, B=net.B; J=net.J; end

G=diag(max(diag(F2-hF1*B'-B*hF1'+B*hE1p*B')./sum(T),eps));

% --- DIAGNOSTIC OUTPUT (Team 2) ---
% Prints exactly what is active during training
if isfield(net, 'reg') && ~isempty(net.reg)
    str_mod = ''; str_l1 = '';
    if isfield(net.reg, 'use_modularity') && net.reg.use_modularity
        if isfield(net.reg, 'tau') && net.reg.tau > 0
            str_mod = sprintf('→ Modularity Active (tau=%.1f) ', net.reg.tau);
        end
    end
    if isfield(net.reg, 'use_L1') && net.reg.use_L1
        if isfield(net.reg, 'rho') && net.reg.rho > 0
            str_l1 = sprintf('→ L1-Sparsity Active (rho=%.4f) ', net.reg.rho);
        end
    end
    if ~isempty(str_mod) || ~isempty(str_l1)
        fprintf('   %s%s\n', str_mod, str_l1);
    end
end
% ----------------------------------

% --- 4. State Parameters (A, W, h, C) ---
I=eye(m); O=ones(m)-I;
if ~fixedC
    Mask=[I;O;ones(1,m);ones(Minp,m)];
    EL=[E3,E4',Zt1,F3';E4,E1,phiZ,F4';Zt1',phiZ',sum(T)-ntr,InpS';F3,F4,InpS,F6_];
    ER=[E2,E5,Zt0,F5_+[Ez(1:m)*Inp{1}(:,1)']]; 
else
    C0=net.C; Mask=[I;O;ones(1,m)];
    EL=[E3,E4',Zt1;E4,E1,phiZ;Zt1',phiZ',sum(T)-ntr];
    ER=[E2-C0*F3,E5-C0*F4,Zt0-C0*InpS];
end

AWhC=zeros(m,size(EL,1)); W=zeros(m);
for i=1:m
    % 4a. Standard Regularization
    if ~isempty(net.reg)
        [Reg_EL, Reg_ER] = getORegularization(net.reg, Mask, net.Sigma, m, i);
        el = EL + Reg_EL; er = ER + Reg_ER;
    else, el = EL; er = ER; end

    % =====================================================================
    % 4b. VECTORIZED MODULARITY PRIOR
    % =====================================================================
    if isfield(net.reg, 'use_modularity') && net.reg.use_modularity && isfield(net.reg, 'modularity_mask')
        base_tau = net.reg.tau; 
        if isempty(base_tau), base_tau = 0; end
        
        if base_tau > 0
            row_mask = net.reg.modularity_mask(i, :); 
            est_indices = 2:m;
            w_cols = [1:i-1, i+1:m];
            
            penalties = row_mask(w_cols);
            bad_conn_idx = find(penalties > 1);
            
            if ~isempty(bad_conn_idx)
                penalty_values = net.Sigma(i,i) * base_tau * (penalties(bad_conn_idx) - 1);
                target_indices = est_indices(bad_conn_idx);
                lin_idx = sub2ind(size(el), target_indices, target_indices);
                
                % --- FIX: Removed transpose on penalty_values ---
                el(lin_idx) = el(lin_idx) + penalty_values;
            end
        end
    end
    
    % 4c. Solve (Ridge)
    theta = er(i,:) * pinv(el);

    % =====================================================================
    % 4d. L1 SPARSITY (Soft Thresholding)
    % =====================================================================
    if isfield(net.reg, 'use_L1') && net.reg.use_L1 && net.reg.rho > 0
        rho = net.reg.rho;
        if m > 1
            w_vals = theta(2:m);
            theta(2:m) = sign(w_vals) .* max(0, abs(w_vals) - rho);
        end
    end
    
    AWhC(i,:) = theta;
    W(i,:)=[AWhC(i,2:i) 0 AWhC(i,i+1:m)];
end

net.A=diag(AWhC(:,1)); net.W=W; net.h=AWhC(:,m+1);
if ~fixedC, net.C=AWhC(:,m+2:end); end
net.B=B; net.J=J; net.Gamma=G;

% Helper (Unchanged)
function [Reg_EL, Reg_ER]=getORegularization(reg,Mask,S0,m,irow)
Reg_ER=0; sigma=unique(diag(S0)); LMask=reg.Lreg; Reg0=0; tau=reg.tau; N=size(Mask,1);
if (~isempty(tau) && tau~=0)
    I=eye(N); tmp=LMask(irow,:); ind=find(tmp==1); II=eye(N); II(ind,ind)=0; O1=I-II; Reg0=sigma*tau.*O1;
end
Reg1=0; lambda=reg.lambda;
if (~isempty(lambda) && lambda~=0)
    I=eye(N); tmp=LMask(irow,:); ind=find(tmp==-1); II=eye(N); II(ind,ind)=0; O1=I-II;     
    O2=zeros(m,N); Aindex=ind(ismember(ind,1:m)); II=eye(length(Aindex)); O2(Aindex,Aindex)=II;
    Reg1=sigma*lambda.*O1; Reg_ER=sigma*lambda*O2;
end
Reg_EL=Reg0+Reg1;