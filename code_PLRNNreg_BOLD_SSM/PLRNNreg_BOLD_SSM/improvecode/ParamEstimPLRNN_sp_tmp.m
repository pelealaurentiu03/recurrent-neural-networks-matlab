function [net,ELL]=ParamEstimPLRNN_sp_tmp(net,Ez,V,Ephizi,Ephizij,Eziphizj,X_,Inp_,rp, fixedS, fixedC, fixedG, fixedB,lam)
if nargin<14, lam=0; end

% implements parameter estimation for PLRNN system
% z_t = A z_t-1 + W max(z_t-1-h,0) + C Inp_t + e_t , e_t ~ N(0,S)
% x_t = B (hrf*z_tau:t) + M rp + nu_t , nu_t ~ N(0,G)
%
% NOTE: X_, Inp_ can be matrices or cell arrays. If provided as cell
% arrays, each cell is interpreted as a single 'trial'; info is aggregated
% across trials, but temporal gaps are assumed between
% trials, i.e. temporal breaks are 'inserted' between trials
%
%
% REQUIRED INPUTS:
% net: network class
% Ez: MxT matrix of state expectancies as returned by StateEstPLRNN
% V: state covariance matrix E[zz']-E[z]E[z]'
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% X_: NxT matrix of observations, or cell array of observation matrices
% Inp_: KxT matrix of external inputs, or cell array of input matrices
% rp: PxT nuiscance variables
%
% OPTIONAL INPUTS:
% fixedS: fixed process noise-cov matrix 
% fixedC: fixed regression matrix for external inputs
% fixedG: fixed observation noise-cov matrix 
% fixedB: fixed observation regression coefficients 
% lam:	  regularization parameter
%
% OUTPUTS:
% net: network class containing updated parameters
% ELL: expected (complete data) log-likelihood


eps=1e-5;  % minimum variance allowed for in S and G

if iscell(X_), X=X_; Inp=Inp_; else X{1}=X_; Inp{1}=Inp_; end;
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
dt=length(net.hrf);                 %GK length of hrf
H=net.getConvolutionMtx;            %GK convolution matrix

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
e1=zeros(m,m,net.n); e4=zeros(m,m,net.n); e5=zeros(m,m,net.n); f4=zeros(Minp,m,net.n);
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
        
        hF1=hF1+X{i}(:,t)*hEzi0(k0)';   %nuiscance variables M
        F7=F7+X{i}(:,t)*rp(t,:);        %nuiscance variables M
        F8=F8+rp(t,:)'*rp(t,:);         %nuiscance variables M
        hF9=hF9+rp(t,:)'*hEzi0(k0)';    %hrf
        
        trial=net.get_trial_curr(t);
        e5(:,:,trial)=e5(:,:,trial)+Eziphizj0(k0,k1);         %trial-dependent W_n
        e4(:,:,trial)=e4(:,:,trial)+Eziphizj0(k1,k1)';
        f4(:,:,trial)=f4(:,:,trial)+Inp{i}(:,t)*Ephizi0(k1)';
        e1(:,:,trial)=e1(:,:,trial)+Ephizij0(k1,k1);
        
        %GK: for E[h(phi(Ezi))h(phi(Ezi'))], using off-diagonal elements: E[zq zq] (for time lags)
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
        tmp=triu(ones(m,m),+1); %to help search elements
        tmp=tmp+tmp';
        [ii,jj]=ind2sub(size(tmp),find(tmp==1));
        for ix=1:length(ii)         %off diagonal elements: E[zqi,zqj]
            qi=ii(ix); qj=jj(ix);
            indcol=qj:m:T*m;
            indrow=qi:m:T*m;
            Ephizijm=Ezizij0g(indrow,indcol);
            Ephizijtt=Ephizijm(tt-dim:tt,tt-dim:tt);
            hhE(qi,qj)=sum(sum(Cd*Ephizijtt*Cdt));
        end
        hE1=hE1+hhE;
        clear hhE
        
        [VH]=getConvolvedCovanceMtx(Ezizi,net,Ezizij0g);
%         hE1_test=VH(tt-dim:tt,tt-dim:tt);
%         hE1_test2=VH(t-dim:t,t-dim:t);      
%        diag(hE1)
%        sum(sum(hE1_test))
%        sum(sum(hE1_test2))
        %  txt=sprintf('hE1=%2.4f, hEtest=%2.4f',hE1,hE1_test);
      %  disp(txt);
        
        keyboard
    end
    E1pkk=E1pkk+Ephizij0(k0,k0);
    E3pkk=E3pkk+Ezizi0(k0,k0);
    
    %fuer t=T
    tt=T;
    Cd=net.getDiagHrf(tt);
    Cdt=fliplr(Cd);
    dim=size(Cd,2)-1;
    for im=1:m %diagnoal hrf components
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
    
end;
E1p=E1+E1pkk;
hE1p=hE1+hE1pkk;
F5=F5_+f5_1;
F6=F6_+f6_1;


% B, M Parameters simultaneously with (xt-BM*o)'G^-1(xt-BM*o)
%---------------------------------------------------------------------------
if ~fixedB
        
    if isempty(net.Bmap)
        bm1=[hF1 F7];
        bm2=[hE1p hF9'; hF9 F8];
        BM=bm1*bm2^-1;
        B=BM(1:net.q,1:net.p);
        M=BM(1:net.q, net.p+1:end);
    else
        Mz=3;
               
        %Bmask=[ones(Nx,Mz), zeros(Nx,Mz); zeros(Nx,Mz) ones(Nx, Mz)];
        Bmask=net.Bmap;
        Mzs=sum(Bmask'); %how many latent dim on each obs
        Mask=[Bmask ones(net.q, net.m)];
        
        bm1=[hF1 F7];
        bm2=[hE1p hF9'; hF9 F8];
        
        B=zeros(net.q, net.p); M=zeros(net.q,net.m);
        for i=1:net.q
            k=find(Mask(i,:));

            X=bm2(k,k);
            Y=bm1(i,k);
            BM=Y*X^-1;  %besser als Struktur; dann koennte B irregulaer sein

            B(i,find(Bmask(i,:)))=BM(1:Mzs(i));
            M(i,:)=BM(Mzs(i)+1:end);
        end
        net.B=B;
        net.M=M;  
    end
    
else
    B=net.B;
    M=net.M;
    %M=(F7-B*hF9')*(F8^-1);
end
%---------------------------------------------------------------------------

% %GK solve for B and M simultaneously
% % B, M Parameters simultaneously solved individually
% m1=(F7*(F8^-1))*(hF9'*(F8^-1))^-1-hF1*(hE1p^-1);
% m2=((hF9'*(F8^-1))^-1-hF9*(hE1p^-1))^-1;
% M2=m1*m2;
% b1=F7*(F8^-1)-(hF1*(hE1p^-1))*(hF9*(hE1p^-1))^-1;
% b2=(hF9'*(F8^-1)-(hF9*(hE1p^-1))^-1)^-1;
% B2=b1*b2;
%B=(hF1-M*hF9)*(hE1p^-1);  %==b1*b2

%%% solve for parameters
% B, M Parameters sequentially
%M=net.M;
%B=(hF1-M*hF9)*(hE1p^-1);
%M=(F7-B*hF9')*(F8^-1);

% Gamma Parameters
%--------------------------------------------------------------------------
% enforce non-negativity explicitly:
movterm=-F7*M'-M*F7'+B*hF9'*M'+M*hF9*B'+M*F8'*M';

if ~fixedG
    G=diag(max(diag(F2-hF1*B'-B*hF1'+B*hE1p'*B'+movterm)./sum(T),eps));
else
    G=net.Gamma;
end
%--------------------------------------------------------------------------


%% solve for
% - interaction weight matrix W
% - auto-regressive weights A
% - bias terms h
% - external regressor weights C
% in one go:
I=eye(m);
O=ones(m)-I;
if  ~fixedC
    Mask=[I;O;ones(1,m);ones(Minp,m)];
    AWhC=zeros(m,m+1+Minp);
    EL=[E3,E4',Zt1,F3';E4,E1+lam*I,phiZ,F4';Zt1',phiZ',Tsum(end)-ntr,InpS';F3,F4,InpS,F6_];
    ER=[E2,E5,Zt0,F5_];
else
    C0=net.C;
    Mask=[I;O;ones(1,m)];
    AWhC=zeros(m,m+1);
    EL=[E3,E4',Zt1;E4,E1+lam*I,phiZ;Zt1',phiZ',Tsum(end)-ntr];
    ER=[E2-C0*F3,E5-C0*F4,Zt0-C0*InpS];
end
W=zeros(m);
for i=1:m
    k=find(Mask(:,i));
    X=EL(k,k);
    Y=ER(i,k);
    AWhC(i,:)=Y*X^-1;
    W(i,:)=[AWhC(i,2:i) 0 AWhC(i,i+1:m)];
end;
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
end;
%--------------------------------------------------------------------------

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
net.M=M;
net.Sigma=S;
net.C=C;
net.h=h;

%get likelihood within class
if nargout > 1
    %Z=reshape(Ez,net.p,net.T);
    %ELL=net.ComplLogLike(Inp{1},X{1},Z,rp);
    
    
    %% compute expected log-likelihood (if desired)
    E3p=E3+E3pkk;
    LL0=0;
    
    LL0=mu0{:}'*S^-1*mu0{:} + mu0{:}'*S^-1*C*Inp{:}(:,1) +  Inp{:}(:,1)'*C'*S^-1*mu0{:} ...
        -mu0{:}'*S^-1*Ez(1:m) -Ez(1:m)'*S^-1*mu0{:};
    
    LL1=trace(S^-1*E3p)-trace(S^-1*A*E2')-trace(S^-1*W*E5')-trace(S^-1*C*F5')-trace(A'*S^-1*E2) ...
        +trace(A'*S^-1*A*E3)+trace(A'*S^-1*W*E4)+trace(A'*S^-1*C*F3) -trace(W'*S^-1*E5)...
        +trace(W'*S^-1*A*E4') +trace(W'*S^-1*W*E1) +trace(W'*S^-1*C*F4)...
        -trace(C'*S^-1*F5)+ trace(C'*S^-1*A*F3')  + trace(C'*S^-1*W*F4')...
        + trace(C'*S^-1*C*F6);  %alle h-Terme fehlen!!!
    
    h_expressions= -Zt0'*S^-1*h -h'*S^-1*Zt0 + Zt1'*A'*S^-1*h + h'*S^-1*A*Zt1 ...
        + phiZ'*W'*S^-1*h + h'*S^-1*W*phiZ + T*h'*S^-1*h + h'*S^-1*C*InpS ...
        + InpS'*C'*S^-1*h;
    LL1=LL1 + h_expressions;
    
    LL2=trace(G^-1*F2)-trace(G^-1*B*hF1') - trace(G^-1*M*F7') ...
        - trace(B'*G^-1*hF1) + trace(B'*G^-1*B*hE1p) - trace(M'*G^-1*F7) ...
        + trace(M'*G^-1*B*hF9') + trace(B'*G^-1*M*hF9) + trace(M'*G^-1*M*F8);
    
    ELL=-1/2*(LL0+LL1+LL2+T*log(det(G))+T*log(det(S)));    
end



% (c) 2016 Durstewitz
% 2017 Koppe adapted
