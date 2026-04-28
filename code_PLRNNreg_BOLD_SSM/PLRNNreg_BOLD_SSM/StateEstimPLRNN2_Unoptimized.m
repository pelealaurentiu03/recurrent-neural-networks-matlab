function [z,U,d,Err]=StateEstimPLRNN2(net,Inp_,X_,rp,z0,d0,tol,eps,flipAll)

% implements state estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1-h,0) + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B (hrf*z_tau:t) + J r_t + nu_t , nu_t ~ N(0,G)
% see Koppe et al 2019, PLOS Computational Biology

% REQUIRED INPUTS:
% network structure 'net' with properties:
    % A: MxM diagonal matrix 
    % W: MxM off-diagonal matrix
    % C: MxK matrix of regression weights multiplying with Inp
    % S: MxM diagonal covariance matrix (Gaussian process noise)
    % mu0_: Mx1 vector of initial values, or cell array of Mx1 vectors
    % B: NxM matrix of regression weights
    % J: NxP matrix of regression weights of nuiscance covariates 
    % G: NxN diagonal covariance matrix
    % h: Mx1 vector of thresholds
   
% X_: NxT matrix of observations, or cell array of observation matrices
% Inp_: KxT matrix of external inputs, or cell array of input matrices 
% rp: TxP matrix of nuiscance covariates (equal to 'R' transpose in manuscript)

% OPTIONAL INPUTS:
% z0: initial guess of state estimates provided as (MxT)x1 vector
% d0: initial guess of constraint settings provided as 1x(MxT) vector 
% tol: acceptable relative tolerance for error increases (default: 1e-2)
% eps: small number added to state covariance matrix for avoiding
%      singularities (default: 0)
% flipAll: flag which determines whether all constraints are flipped at
%          once on each iteration (=true) or whether only the most violating
%          constraint is flipped on each iteration (=false) 
% (Note: option not implemented yet)

% OUTPUTS:
% z: estimated state expectations
% U: -Hessian
% Err: final total threshold violation error
% d: final set of constraints (ie, for which z>h) 
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

if nargin<7, tol=1e-2; end
if nargin<8, eps=[]; end
if nargin<9, flipAll=false; end

m=length(A);   % # of latent states

if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_;
else X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; end
ntr=length(X);  % # of distinct trials

%%% construct block-banded components of Hessian U0, U1, U2, U3 and vectors
%%% v0, v1
%(see eq. 6 and beyond in Koppe et al 2019, PLoS CB)
%--------------------------------------------------------------------------
u0=S^-1+A'*S^-1*A; K0=-A'*S^-1;
u2a=W'*S^-1*W;           
u2b=B'*G^-1*B;

u1=W'*S^-1*A; K2=-W'*S^-1;

U0=[]; U2a=[]; U2b=[]; U1=[];
v0=[]; v1a=[];  v1b=[]; 
Tsum=0;
for i=1:ntr   
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
    vkb=-W'*S^-1*I(:,2:T);
    v1a=([vkb(1:end) zeros(1,m)])';    
   
    vka=B'*G^-1*X{i};  vr1=B'*G^-1*J*rp'; 
    vr1=reshape(vr1,numel(vr1),1);   
    v1b=(vka(1:end))';      
    v1b=v1b-vr1;
end

% ---------------------------------------------------------------------
%initialize states and constraint vector
n=1; idx=0; k=[];
if nargin>4 && ~isempty(z0), z=z0(1:end)'; else z=randn(m*Tsum,1); end    
if nargin>5 && ~isempty(d0), d=d0; else d=zeros(1,m*Tsum); d(z>0)=1; end     
Err=1e16;
y=rand(m*Tsum,1); LL=d*y; % define arbitrary projection vector for detecting already visited solutions 
U=[]; dErr=-1e8;

% mode search iterations
while ~isempty(idx) && isempty(k) && dErr<tol*Err(n)
    % iterate as long as not all constraints are satisfied (idx), the
    % current solution has not already been visited (k), and the change in
    % error (dErr) remains below the tolerance level 
    
    % save last step
    zsv=z; Usv=U; dsv=d;
    if n>1
        if flipAll, dsv(idx)=1-d(idx);
        else dsv(idx(r))=1-d(idx(r)); end
    end
    
    % (1) solve for states Z given constraints d
    D=spdiags(d',0,m*Tsum,m*Tsum);   
    h1=d; h2=[zeros(1,m) h1(1:(Tsum-1)*m)];         

    A1=U0+D*U2a*D'+(D*U1)+(D*U1)';  
    A2=H'*U2b*H;                    
    U=A1+1/2*(A2+A2');
   
    if ~isempty(eps), U=U+eps*speye(size(U)); end   % avoid singularities   
    B1=(v0+d'.*v1a); 
    B2=(H'*v1b);             
    z=U\(B1+B2);

    % (2) flip violated constraint(s)
    idx=find(abs(d-(z>0)'));
    ae=abs(z(idx));
    n=n+1; Err(n)=sum(ae); dErr=Err(n)-Err(n-1);
    if flipAll, d(idx)=1-d(idx);                    % flip all constraints at once
    else [~,r]=max(ae); d(idx(r))=1-d(idx(r)); end  % flip constraints only one-by-one

    % terminate when revisiting already visited edges:
    l=d*y; k=find(LL==l); LL=[LL l]; 
end

if dErr<tol*Err(n)
    % if idx=[] or k!=[], display final error & change in error
    z=reshape(z,m,Tsum);
    if flipAll, d(idx)=1-d(idx);
    else d(idx(r))=1-d(idx(r)); end
    disp(['#1 - ' num2str(dErr) '   ' num2str(Err(end))])
    Err=Err(2:end);
else
    % if dErr exceeded tolerance, display # of still violated constraints
    z=reshape(zsv,m,Tsum);
    U=Usv;
    d=dsv;
    Err=Err(2:end-1);
    disp(['#2 - ' num2str(length(idx)) '   ' num2str(length(k))])
end

% (c) 2016 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
% adapted 2019, Georgia Koppe, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
