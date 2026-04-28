function net=Our_ParamEstimPLRNN_sp(net,Ez,V,Ephizi,Ephizij,Eziphizj,X_,Inp_,rp, fixedS, fixedC, fixedG, fixedB,lam)
% =========================================================================
% MODIFIED FILE: Our_ParamEstimPLRNN_sp.m
% TEAM 2 CONTRIBUTION: 
% Modified by Andrei on 18.11.2025 - Added L1 (Lasso) Regularization for Sparsity on W
% =========================================================================
% implements parameter estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1-h,0) + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B (hrf*z_tau:t) + J r_t + nu_t , nu_t ~ N(0,G)
eps=1e-5;  % minimum variance allowed for in S and G
if iscell(X_), X=X_; Inp=Inp_; else X{1}=X_; Inp{1}=Inp_; end
ntr=length(X);
m=size(Ez,1);
N=size(X{1},1);
Minp=size(Inp{1},1);
T=cell2mat(cellfun(@size,X,'UniformOutput',false)'); T=T(:,2);
Tsum=cumsum([0 T']);
Lsum=Tsum.*m;
%% compute E[zz'] from state cov matrix V
Ez=Ez(1:end)';
Ezizi=sparse(m*sum(T),m*sum(T));
dt=length(net.hrf);                 %length of hrf
H=net.getConvolutionMtx;            %convolution matrix
for i=1:ntr
    for t=Tsum(i)+1:(Tsum(i+1)-1)
        k0=(t-1)*m+1:t*m;
        if (t+dt)*m>Lsum(2)        
            k1=t*m+1:Lsum(2);
        else
            k1=t*m+1:(t+dt)*m; %dt time steps ahead
        end
        Ezizi(k0,[k0 k1])=V(k0,[k0 k1])+Ez(k0)*Ez([k0 k1])';
        Ezizi(k1,k0)=Ezizi(k0,k1)';
    end
    Ezizi(k1,k1)=V(k1,k1)+Ez(k1)*Ez(k1)';
end
%% compute all expectancy sums across trials & time points (eq. 16)
E1=zeros(m); E2=E1; E3=E1; E4=E1; E5=E1; E1pkk=E1; E3pkk=E1; E11=E1;
F1=zeros(N,m); F2=zeros(N,N); F3=zeros(Minp,m); F4=F3; F5_=zeros(m,Minp); F6_=zeros(Minp,Minp);
hF1=zeros(N,m); hE1pkk=E1pkk; hE1=E1;
F7=zeros(N,size(rp,2));             %sum_t x_t rp_tT
F8=zeros(size(rp,2),size(rp,2));    %sum_t r_t rp_tT
hF9=zeros(size(rp,2),m);
%for potential trial-dependent sum terms of W_n
ntrials=1; 
e1=zeros(m,m,ntrials); e4=zeros(m,m,ntrials); e5=zeros(m,m,ntrials); f4=zeros(Minp,m,ntrials);
f5_1=F5_; f6_1=F6_;
Zt1=zeros(m,1); Zt0=zeros(m,1); phiZ=zeros(m,1); InpS=zeros(Minp,1);
for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    Ephizi0=Ephizi(mt);
    Ezizi0=Ezizi(mt,mt);
    Ephizij0=Ephizij(mt,mt);
    Eziphizj0=Eziphizj(mt,mt);
    F1=F1+X{i}(:,1)*Ephizi0(1:m)';
    F2=F2+X{i}(:,1)*X{i}(:,1)';
    f5_1=f5_1+Ez0(1:m)*Inp{i}(:,1)';
    f6_1=f6_1+Inp{i}(:,1)*Inp{i}(:,1)';
    
    hEzi0=H*Ez(mt);                     %hrf convolved states
    Ezizij0g=Ezizi(mt,mt);              %for later hrf convolution
    F7=F7+X{i}(:,1)*rp(1,:);            %nuiscance variable term
    F8=F8+rp(1,:)'*rp(1,:);             %nuiscance variable term
    hF9=hF9+rp(1,:)'*hEzi0(1:m)';
    hF1=hF1+X{i}(:,1)*hEzi0(1:m)';      %hrf term
    
    for t=2:T(i)
        k0=(t-1)*m+1:t*m;       % t
        k1=(t-2)*m+1:(t-1)*m;   % t-1
        E1=E1+Ephizij0(k1,k1);
        E2=E2+Ezizi0(k0,k1);
        E3=E3+Ezizi0(k1,k1);
        E4=E4+Eziphizj0(k1,k1)';
        E5=E5+Eziphizj0(k0,k1);
        F1=F1+X{i}(:,t)*Ephizi0(k0)';
        
        F2=F2+X{i}(:,t)*X{i}(:,t)';
        F3=F3+Inp{i}(:,t)*Ez0(k1)';
        F4=F4+Inp{i}(:,t)*Ephizi0(k1)';
        F5_=F5_+Ez0(k0)*Inp{i}(:,t)';
        F6_=F6_+Inp{i}(:,t)*Inp{i}(:,t)';
        
        hF1=hF1+X{i}(:,t)*hEzi0(k0)';   %nuiscance variables 
        F7=F7+X{i}(:,t)*rp(t,:);        %nuiscance variables 
        F8=F8+rp(t,:)'*rp(t,:);         %nuiscance variables 
        hF9=hF9+rp(t,:)'*hEzi0(k0)';    %hrf
        
        % Hrf calculation block (standard)
        tt=t-1;
        Cd=net.getDiagHrf(tt);
        Cdt=fliplr(Cd);
        dim=size(Cd,2)-1;
        for im=1:m  %diagonal hrf components
            qi=im:m:T*m; qj=im:m:T*m;
            Ephizijm=Ezizij0g(qi,qj);
            Ephizijtt=Ephizijm(tt-dim:tt,tt-dim:tt);
            hhE(im,im)=sum(sum(Cd*Ephizijtt*Cdt));
        end
        tmp=triu(ones(m,m),+1); 
        tmp=tmp+tmp';
        [ii,jj]=ind2sub(size(tmp),find(tmp==1));
        for ix=1:length(ii)         %off diagonal elements
            qi=ii(ix); qj=jj(ix);
            indcol=qj:m:T*m;
            indrow=qi:m:T*m;
            Ephizijm=Ezizij0g(indrow,indcol);
            Ephizijtt=Ephizijm(tt-dim:tt,tt-dim:tt);
            hhE(qi,qj)=sum(sum(Cd*Ephizijtt*Cdt));
        end
        hE1=hE1+hhE;
        clear hhE
    end
    E1pkk=E1pkk+Ephizij0(k0,k0);
    E3pkk=E3pkk+Ezizi0(k0,k0);
    
    %for t=T block
    tt=T;
    Cd=net.getDiagHrf(tt);
    Cdt=fliplr(Cd);
    dim=size(Cd,2)-1;
    for im=1:m 
        qi=im:m:T*m; qj=im:m:T*m;
        Ephizijm=Ezizij0g(qi,qj);
        Ephizijtt=Ephizijm(tt-dim:tt,tt-dim:tt);
        hhE(im,im)=sum(sum(Cd*Ephizijtt*Cdt));
    end
    tmp=triu(ones(m,m),+1); tmp=tmp+tmp';
    [ii,jj]=ind2sub(size(tmp),find(tmp==1));
    for ix=1:length(ii)
        qi=ii(ix); qj=jj(ix);
        indcol=qj:m:T*m; indrow=qi:m:T*m;
        Ephizijm=Ezizij0g(indrow,indcol);
        Ephizijtt=Ephizijm(tt-dim:tt,tt-dim:tt);
        hhE(qi,qj)=sum(sum(Cd*Ephizijtt*Cdt));
    end
    hE1pkk=hE1pkk+hhE;
    
    zz=reshape(Ez0,m,T(i))';
    Zt1=Zt1+sum(zz(1:end-1,:))';
    Zt0=Zt0+sum(zz(2:end,:))';
    pz=reshape(Ephizi0,m,T(i))';
    phiZ=phiZ+sum(pz(1:end-1,:))';
    InpS=InpS+sum(Inp{i}(:,2:end)')';
    
end
E1p=E1+E1pkk;
hE1p=hE1+hE1pkk;
F5=F5_+f5_1;
F6=F6_+f6_1;
% B, J Parameters (Standard Observation Model - No changes here)
%---------------------------------------------------------------------------
if ~fixedB 
    if isempty(net.Bmap) 
        bm1=[hF1 F7];
        bm2=[hE1p hF9'; hF9 F8];
        BM=bm1*bm2^-1;
        B=BM(1:net.q,1:net.p);
        J=BM(1:net.q, net.p+1:end);
    else                
        Bmask=net.Bmap;
        Mzs=sum(Bmask'); 
        Mask=[Bmask ones(net.q, net.m)];
        bm1=[hF1 F7];
        bm2=[hE1p hF9'; hF9 F8];
        B=zeros(net.q, net.p); J=zeros(net.q,net.m);
        for i=1:net.q
            k=find(Mask(i,:));
            X=bm2(k,k);
            Y=bm1(i,k);
            BM=Y*X^-1;  
            B(i,find(Bmask(i,:)))=BM(1:Mzs(i));
            J(i,:)=BM(Mzs(i)+1:end);
        end
        net.B=B;
        net.J=J;  
    end
else 
    B=net.B;
    J=net.J;
end
% Gamma Parameters
%--------------------------------------------------------------------------
movterm=-F7*J'-J*F7'+B*hF9'*J'+J*hF9*B'+J*F8'*J';
if ~fixedG
    G=diag(max(diag(F2-hF1*B'-B*hF1'+B*hE1p'*B'+movterm)./sum(T),eps));
else
    G=net.Gamma;
end
%% solve for A, W, h, C
% =========================================================================
% TEAM 2 CONTRIBUTION: IMPLEMENTING L1 SPARSITY (LASSO) ON W
% =========================================================================
I=eye(m);
O=ones(m)-I;
if  ~fixedC
    Mask=[I;O;ones(1,m);ones(Minp,m)];
    AWhC=zeros(m,m+1+Minp);
    EL=[E3,E4',Zt1,F3';E4,E1,phiZ,F4';Zt1',phiZ',Tsum(end)-ntr,InpS';F3,F4,InpS,F6_];
    ER=[E2,E5,Zt0,F5_];
else
    C0=net.C;
    Mask=[I;O;ones(1,m)];
    AWhC=zeros(m,m+1);
    EL=[E3,E4',Zt1;E4,E1,phiZ;Zt1',phiZ',Tsum(end)-ntr];
    ER=[E2-C0*F3,E5-C0*F4,Zt0-C0*InpS];
end
W=zeros(m);
% Loop through each row of the state equation
for i=1:m
    
    % 1. Apply Standard L2 Regularization (Koppe et al. original)
    if ~isempty(net.reg) 
        [Reg_EL, Reg_ER] = getORegularization(net.reg, Mask, net.Sigma, m, i);
        el = EL + Reg_EL;
        er = ER + Reg_ER;
    else
        el = EL;
        er = ER;
    end
    
    % 2. Compute the "Ridge" (Standard) Solution first
    k=find(Mask(:,i));
    X_mat=el(k,k);
    Y_vec=er(i,k);
    
    % Direct Matrix Inversion (Least Squares/Ridge)
    % This gives the un-sparse, standard estimate
    theta_ols = Y_vec * inv(X_mat);
    
    % Store it initially
    AWhC(i,:) = theta_ols;

    % =====================================================================
    % TEAM 2 CONTRIBUTION: APPLY SOFT-THRESHOLDING FOR SPARSITY
    % This mathematically forces small weights to become EXACTLY zero.
    % =====================================================================
    if isfield(net.reg, 'use_L1') && net.reg.use_L1 && net.reg.rho > 0
        rho = net.reg.rho;
        
        % The "off-diagonal W" parameters are located at indices 2:m 
        % of the packed AWhC vector for row i.
        % (Index 1 is A_i, Index m+1 is h_i)
        
        if m > 1
            w_inds = 2:m; 
            w_vals = AWhC(i, w_inds);
            
            % --- SOFT THRESHOLDING OPERATOR ---
            % Formula: sign(w) * max(0, |w| - rho)
            w_sparse = sign(w_vals) .* max(0, abs(w_vals) - rho);
            
            % Overwrite the parameter vector with sparse values
            AWhC(i, w_inds) = w_sparse;
        end
    end
    % =====================================================================
    
    % 3. Reconstruct W matrix from the row
    W(i,:)=[AWhC(i,2:i) 0 AWhC(i,i+1:m)];  
end
A=diag(AWhC(:,1));
h=AWhC(:,m+1);
if ~fixedC
    C=AWhC(:,m+2:end);
else
    C=net.C;
end
% solve for trial-specific parameters mu0
%--------------------------------------------------------------------------
for i=1:ntr
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    mu0{i}=Ez0(1:m)-C*Inp{i}(:,1);
end
%log away in network class
if ~fixedS
    disp('add estimation of Sigma');
else
    S=net.Sigma;    %don't estimate Sigma
end
%collect all parameters in network class
net.mu0=mu0{:};
net.A=A;
net.B=B;
net.W=W;
net.Gamma=G;
net.J=J;
net.Sigma=S;
net.C=C;
net.h=h;
% Helper function for L2 (Original code)
function [Reg_EL, Reg_ER]=getORegularization(reg,Mask,S0,m,irow)
Reg_ER=0;
sigma=unique(diag(S0));
LMask=reg.Lreg;
Reg0=0;
tau=reg.tau;
N=size(Mask,1);
if (~isempty(tau) && tau~=0)
    I=eye(N);
    tmp=LMask(irow,:);
    ind=find(tmp==1); 
    II=eye(N);
    II(ind,ind)=0;
    O1=I-II;    
    Reg0=sigma*tau.*O1;
end
Reg1=0;
lambda=reg.lambda;
if (~isempty(lambda) && lambda~=0)
    I=eye(N);
    tmp=LMask(irow,:);
    ind=find(tmp==-1); 
    II=eye(N);
    II(ind,ind)=0;
    O1=I-II;     
    O2=zeros(m,N);
    Aindex=ind(ismember(ind,1:m));
    II=eye(length(Aindex));
    O2(Aindex,Aindex)=II;
    Reg1=sigma*lambda.*O1;
    Reg_ER=sigma*lambda*O2;
end
Reg_EL=Reg0+Reg1;