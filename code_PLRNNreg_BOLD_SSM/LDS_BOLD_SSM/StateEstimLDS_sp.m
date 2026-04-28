function [z,V]=StateEstimLDS_sp(net,Inp_,X_,rp,eps)

% implements state estimation for PLRNN system
% z_t = A z_t-1 + W z_t-1 + h + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B h(z_t) + J r_t + nu_t , nu_t ~ N(0,G)

% REQUIRED INPUTS:
% net: class containing model parameters as elements:
    % net.A: MxM diagonal matrix 
    % net.W: MxM off-diagonal matrix
    % net.C: MxK matrix of regression weights multiplying with Inp
    % net.h: Mx1 vector of thresholds
    % net.S: MxM diagonal covariance matrix (Gaussian process noise)
    % net.mu0: Mx1 vector of initial values, or cell array of Mx1 vectors
    % net.B: NxM matrix of regression weights
    % net.J: NxP matrix of regression weights of nuiscance covariates
    % net.G: NxN diagonal covariance matrix
% Inp_: KxT matrix of external inputs, or cell array of input matrices 
% X_: NxT matrix of observations, or cell array of observation matrices

% OPTIONAL INPUTS:
% eps: small number added to state covariance matrix for avoiding
%      singularities (default: 0)

% OUTPUTS:
% z: estimated state expectations
% V: covariance matrix
%--------------------------------------------------------------------------

%get parameters from network class
A=net.A;
W=net.W;
S=net.Sigma;
mu0_=net.mu0;
J=net.J;
B=net.B;
G=net.Gamma;
C=net.C;
h=net.h;
H=net.getConvolutionMtx;

m=length(A);   % # of latent states
if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_;
else X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; end
ntr=length(X);  % # of distinct trials

%%% construct block-banded components of Hessian U0, U1, U2, U3, and 
% vectors/ matrices v0, v1, as specified in the objective 
% function Q(Z)
%--------------------------------------------------------------------------
u0=S^-1+A'*S^-1*A; K0=-A'*S^-1;
u2a=W(:,:,1)'*S^-1*W(:,:,1);            
u2b=B'*G^-1*B;
u1=W(:,:,1)'*S^-1*A; K2=-W(:,:,1)'*S^-1;

U0=[]; U2a=[]; U2b=[]; U1=[];
v0=[]; v1a=[];  v1b=[]; 
Tsum=0;
for i=1:ntr   % acknowledge trial breaks
    T=size(X{i},2); Tsum=Tsum+T;

    U0_ = repBlkDiag(u0,T); KK0 = repBlkDiag(K0,T); 
    U2a_ = repBlkDiag(u2a,T); 
    U2b_ = repBlkDiag(u2b,T); 
    U1_ = repBlkDiag(u1,T); 
    KK2 = repBlkDiag(K2,T);
    
    KK0=blkdiag(KK0,K0);
    kk=(T-1)*m+1:T*m; U0_(kk,kk)=S^-1;
    U0_=U0_+KK0(m+1:end,1:T*m);     
    KK0=KK0'; U0_=U0_+KK0(1:T*m,m+1:end);
    U2a_(kk,kk)=zeros(m,m);
    U2b_(kk,kk)=B'*G^-1*B;                   
    U1_(kk,kk)=0; KK2=blkdiag(KK2,K2);      
    U1_=U1_+KK2(m+1:end,1:T*m);
    U0=sparse(U0_); U2a=sparse(U2a_);  U2b=sparse(U2b_); U1=sparse(U1_);

    I=C*Inp{i}+repmat(h,1,T);
    vka=S^-1*I; vka(:,1)=vka(:,1)+S^-1*(mu0{i}-h);     
    vkb=A'*S^-1*I(:,2:T);
    v0=(vka(1:end)-[vkb(1:end) zeros(1,m)])';  
    vkb=[];
    for t=2:T
        trial=net.get_trial_curr(t);                 
        vkb=[vkb -W(:,:,trial)'*S^-1*I(:,t)];
    end
    v1a=([vkb(1:end) zeros(1,m)])';    
   
    vka=B'*G^-1*X{i};  vr1=B'*G^-1*J*rp'; vr1=reshape(vr1,numel(vr1),1);   
    v1b=(vka(1:end))';      
    v1b=v1b-vr1;
end

A1=U0+U2a+U1+U1';
A2=H'*U2b*H;
U=A1+1/2*(A2+A2');
if ~isempty(eps), U=U+eps*speye(size(U)); end  % avoid singularities
B1=(v0+v1a);
B2=(H'*v1b);
z=U\(B1+B2);
z=reshape(z,m,Tsum);

% ----- compute block-tridiag cov mtx ---------------------------------
eps2=1e-9;
U0=zeros(m*Tsum,2*m);
for t=1:Tsum
    k0=(t-1)*m+1:t*m;
    k1=max(0,(t-2)*m)+1:t*m;
    U0(k0,1:2*m)=[zeros(m,(t<2)*m) U(k0,k1)];
end
hrf=net.hrf; 
dt=length(hrf); 
clear hrf; 
V0=invblocktridiag(U0,m,dt);

V=sparse(m*T,m*T);
v=V0(:,dt*m+1:end);
%sort diagonal elements of V
for t=1:T
    k0=(t-1)*m+1:(t*m);
    V(k0,k0)=v(k0,1:m);
end
%sort off diagonal elements of V
for d=1:dt
    drow=m*d+1; dcol=1+(dt-d)*m;
    v=V0(drow:end,dcol:dcol+m-1);
    for t=1:T-d
        k0=(t-1)*m+1:(t*m);
        k1=(t-1)*m+1:(t*m); k1=k1+m*d;
        V(k0,k1)=v(k0,1:m)';
        V(k1,k0)=v(k0,1:m); %symmetry (not necessary, see next line)
    end
end

% ensure proper covariance matrix
V=(V+V')./2;    % ensure symmetry
for i=1:m*Tsum, V(i,i)=max(V(i,i),eps2); end

% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
% edits 2017, Georgia Koppe
