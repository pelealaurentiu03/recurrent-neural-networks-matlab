function net=ParamEstimLDS_sp(net,Ez,V,X_,Inp_,rp, fixedS, fixedC, fixedG, fixedB)

eps=1e-5;
H=net.getConvolutionMtx;

if iscell(X_), X=X_; Inp=Inp_; else X{1}=X_; Inp{1}=Inp_; end
ntr=length(X);
m=size(Ez,1);
N=size(X{1},1);
T=cell2mat(cellfun(@size,X,'UniformOutput',false)'); T=T(:,2);
Tsum=cumsum([0 T']);
Lsum=Tsum.*m;

Ez=Ez(1:end)';
Ezizi=sparse(m*sum(T),m*sum(T));
hrf=net.hrf; dt=length(hrf); clear hrf
for i=1:ntr
    for t=Tsum(i)+1:(Tsum(i+1)-1)
        k0=(t-1)*m+1:t*m;
        if (t+dt)*m>Lsum(2)
            k1=t*m+1:Lsum(2);
        else
            k1=t*m+1:(t+dt)*m; 
        end
        Ezizi(k0,[k0 k1])=V(k0,[k0 k1])+Ez(k0)*Ez([k0 k1])';
        Ezizi(k1,k0)=Ezizi(k0,k1)';
    end
    Ezizi(k1,k1)=V(k1,k1)+Ez(k1)*Ez(k1)';
end

Minp=size(Inp{1},1);
%% compute all expectancy sums across trials & time points (eq. 16)
E1=zeros(m); E2=E1; E3=E1; E4=E1; E5=E1; E1pkk=E1; E3pkk=E1; E11=E1;
F1=zeros(N,m); F2=zeros(N,N); F3=zeros(Minp,m); F4=F3; F5_=zeros(m,Minp); F6_=zeros(Minp,Minp);
hF1=zeros(N,m); hE1pkk=E1pkk; hE1=E1; 

F7=zeros(N,size(rp,2));             %sum_t x_t rp_tT
F8=zeros(size(rp,2),size(rp,2));    %sum_t r_t rp_tT
hF9=zeros(size(rp,2),m);            

ntrials=1;
e1=zeros(m,m,ntrials); e4=zeros(m,m,ntrials); e5=zeros(m,m,ntrials); f4=zeros(Minp,m,ntrials); g3=zeros(m,1,ntrials);
% %trial-dependent sum terms for W_t (not implemented here, net.n=trials)
f5_1=F5_; f6_1=F6_; 
Zt1=zeros(m,1); Zt0=zeros(m,1); phiZ=zeros(m,1); InpS=zeros(Minp,1);

for i=1:ntr    
    mt=(Lsum(i)+1:Lsum(i+1))';
    Ez0=Ez(mt);
    Ephizi0=Ez0;	

    Ezizi0=Ezizi(mt,mt);
    Ephizij0=Ezizi0;

    Eziphizj0=Ezizi0;
    F1=F1+X{i}(:,1)*Ephizi0(1:m)';
    F2=F2+X{i}(:,1)*X{i}(:,1)';    
    f5_1=f5_1+Ez0(1:m)*Inp{i}(:,1)';
    f6_1=f6_1+Inp{i}(:,1)*Inp{i}(:,1)';
    
    hEzi0=H*Ez(mt);                     
    Ezizij0g=Ezizi(mt,mt);              
    F7=F7+X{i}(:,1)*rp(1,:);            
    F8=F8+rp(1,:)'*rp(1,:);            
    hF9=hF9+rp(1,:)'*hEzi0(1:m)';
    hF1=hF1+X{i}(:,1)*hEzi0(1:m)';      
    
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
        
        hF1=hF1+X{i}(:,t)*hEzi0(k0)';   %nuiscance variables J
        F7=F7+X{i}(:,t)*rp(t,:);        %nuiscance variables J
        F8=F8+rp(t,:)'*rp(t,:);         %nuiscance variables J
        hF9=hF9+rp(t,:)'*hEzi0(k0)';    %hrf
        
        trial=net.get_trial_curr(t);
        e5(:,:,trial)=e5(:,:,trial)+Eziphizj0(k0,k1);         %trial-dependent W_n
        e4(:,:,trial)=e4(:,:,trial)+Eziphizj0(k1,k1)';
        f4(:,:,trial)=f4(:,:,trial)+Inp{i}(:,t)*Ephizi0(k1)';
        e1(:,:,trial)=e1(:,:,trial)+Ephizij0(k1,k1);

        g3(:,:,trial)=g3(:,:,trial)+Ephizi0(k1);   %for threshold term (phiZ)
        

        % for E[h(phi(Ezi))h(phi(Ezi'))], using off-diagonal elements: E[zq zq] (for time lags)
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
    end
    E1pkk=E1pkk+Ephizij0(k0,k0);
    
    %for t=T
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
end
E1p=E1+E1pkk;
hE1p=hE1+hE1pkk;
F5=F5_+f5_1;
F6=F6_+f6_1;

% solve for B, J parameters simultaneously 
%---------------------------------------------------------------------------
if ~fixedB
    bm1=[hF1 F7];
    bm2=[hE1p hF9'; hF9 F8];
    BM=bm1*bm2^-1;
    B=BM(1:net.q,1:net.p);
    J=BM(1:net.q, net.p+1:end);
else
    B=net.B;
    J=(F7-B*hF9')*(F8^-1);
end

% solve for Gamma parameters
%--------------------------------------------------------------------------
% enforce non-negativity explicitly:
movterm=-F7*J'-J*F7'+B*hF9'*J'+J*hF9*B'+J*F8'*J';

if ~fixedG
    G=diag(max(diag(F2-hF1*B'-B*hF1'+B*hE1p*B'+movterm)./sum(T),eps));   
else
    G=net.Gamma;
end

% solve for 
% - interaction weight matrices Wi
% - auto-regressive weights A
% - bias terms h
% - external regressor weights C
% in one go:
%--------------------------------------------------------------------------
I=eye(m);
O=ones(m)-I;
if ~fixedC
    Mask=[I repmat(O,1,ntrials) ones(m,1) ones(m,Minp)]';
    AWhC=zeros(m,1 +(m-1)*ntrials +1 +Minp);
  
    ee4T=[]; gg3T=[]; ff4=[]; ee5=[];
    for i=1:ntrials
        ee4T=[ee4T e4(:,:,i)'];
        gg3T=[gg3T g3(:,:,i)'];
        ff4=[ff4 f4(:,:,i)];
        ee5=[ee5 e5(:,:,i)];
    end
    %EL first term
    EL1=[E3,ee4T,Zt1,F3'];
    
    %EL mid terms
    ELmid=[];
    K=zeros(m*ntrials,m*ntrials);
    for i=1:ntrials
        k=(i*m-(m-1)):i*m;
        K(k,k)=e1(:,:,i);
        ELmid=[ELmid; e4(:,:,i),K(k,:), g3(:,:,i),f4(:,:,i)'];
    end
    EL2last=[Zt1',gg3T,Tsum(end)-ntr,InpS'];
    ELlast=[F3, ff4, InpS, F6_];
    EL=[EL1; ELmid; EL2last; ELlast];
    ER=[E2,ee5,Zt0, F5_];

else
    C0=net.C;
    Mask=[I;O;ones(1,m)];
    AWhC=zeros(m,m+1);
    EL=[E3,E4',Zt1;E4,E1,phiZ;Zt1',phiZ',Tsum(end)-ntr];
    ER=[E2-C0*F3,E5-C0*F4,Zt0-C0*InpS];
end    
for i=1:m
    k=find(Mask(:,i));
    XX=EL(k,k);
    Y=ER(i,k);
    AWhC(i,:)=Y*XX^-1;
end
A=diag(AWhC(:,1));
h=AWhC(:,1+(m-1)*ntrials+1);
wind=2:(m-1)*ntrials+1;
%fill Wi's
Ws=AWhC(:,wind); n=1;
for i=1:ntrials
    k1=n:(i*(m-1));
    Wi=zeros(m,m);
    for j=1:m
        k2=setdiff(1:m,j);
        Wi(j,k2)=Ws(j,k1);
    end
    W(:,:,i)=Wi;
    n=n+(m-1);
end
if ~fixedC
    C=AWhC(:,n+2:end);
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

% (c) 2016 Durstewitz
% 2017 Koppe adapted
