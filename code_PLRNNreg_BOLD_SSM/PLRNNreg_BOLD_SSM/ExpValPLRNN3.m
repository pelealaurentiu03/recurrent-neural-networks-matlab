function [Ephizi,Ephizij,Eziphizj,V]=ExpValPLRNN3(net,Ez,U)

% please cite (& consult for further details): Durstewitz (2017) PLoS Comp Biol
% 
% computes expectancies E[phi(z)], E[z phi(z)'], E[phi(z) phi(z)'],  
% as given in eqn. 10-15, based on provided state expectancies and Hessian
% 
% REQUIRED INPUTS:
% net: network class instance
% Ez: MxT matrix of state expectancies as returned by StateEstPLRNN
% U: negative Hessian of log-likelihood returned by StateEstPLRNN 
%
% OUTPUTS:
% Ephizi: E[phi(z)]
% Ephizij: E[phi(z) phi(z)']
% Eziphizj: E[z phi(z)']
% V: state covariance matrix E[zz']-E[z]E[z]'

eps=1e-3 ; %minimum enforced variance/ eigenvalue
dt=length(net.hrf)+1;  % length of hrf (i.e. number of time steps back)

[m,T]=size(Ez);

%% invert block-tridiagonal neg. Hessian U
U0=zeros(m*T,2*m);
for t=1:T
    k0=(t-1)*m+1:t*m;
    k1=max(0,(t-2)*m)+1:t*m;
    U0(k0,1:2*m)=[zeros(m,(t<2)*m) U(k0,k1)];
end
V0=invblocktridiag(U0,m,dt);
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
%for i=1:m*T, V(i,i)=max(V(i,i),eps); end;   % restrict min var
[U,E]=eig(full(V),'nobalance'); U=sparse(U); E=sparse(E);
if min(spdiags(E))<eps  % ensure positive-definiteness
    %E=diag(diag(E)-min(diag(E))+eps);
    %V=U'*E*U;
    %V=(V+V')./2;
       
    %get nearest V in terms of Frobenius norm, Copyright (c) 2013, John D'Errico
    niter=100; %set as high as possible, but may slow down code
    V = nearestSPD(V,niter);
end


Ez=Ez(1:end)';
v=spdiags(V,0);
s=sqrt(v);
fk=normpdf(zeros(size(Ez)),Ez,s+zeros(size(Ez)));
Fk=1-normcdf(zeros(size(Ez)),Ez,s+zeros(size(Ez)));

%% E[phi(zi)]
vfk=v.*fk;
Ephizi=vfk+Ez.*Fk;

%% E[phi(zi)*phi(zj)]
Ephizij=sparse(m*T,m*T);
EzzV=sparse(m*T,m*T);
o1=ones(m*2,1);
for t=1:T-1
    
    %dt time steps ahead
    if (t+dt)*m>m*T
        k0=(t-1)*m+1:m*T;
        k1=t*m+1:m*T;
        o1=ones(length(k0),1);
    else
        k0=(t-1)*m+1:(t+dt)*m;
        k1=t*m+1:(t+dt+1)*m;
        o1=ones(length(k0),1); %GK
    end
    if k1(end)>m*T, k1=t*m+1:m*T; end
    
    lam_l=(o1*v(k0)')./(v(k0)*v(k0)'-V(k0,k0).*V(k0,k0));
    lam_l_inv=1./lam_l;
    mu_kl=o1*Ez(k0)'-V(k0,k0).*((Ez(k0)./v(k0))*o1');
    Flk=1-normcdf(zeros(length(k0)),mu_kl',sqrt(lam_l_inv));
    Fkl=1-normcdf(zeros(length(k0)),o1*Ez(k0)',sqrt(lam_l_inv)');
    Nlk=normpdf(zeros(length(k0)),mu_kl',sqrt(lam_l_inv));

    EzzV(k0,k0)=Ez(k0)*Ez(k0)'+V(k0,k0);
    Ephizij(k0,k0)=lam_l_inv'.*(o1*fk(k0)').*(Nlk./lam_l+mu_kl'.*Flk)+ ...
        (v(k0)*Ez(k0)'.*(fk(k0)*o1')+EzzV(k0,k0).*(Fk(k0)*o1')).*Fkl;
end
Ephizij=(Ephizij+Ephizij')./2;

%% E[zi*phi(zj)]
Eziphizj=sparse(m*T,m*T);
for t=1:T-1
    k0=(t-1)*m+1:(t+1)*m;
    Eziphizj(k0,k0)=EzzV(k0,k0).*(o1*Fk(k0)')+Ez(k0)*vfk(k0)';
end

%% E[phi(zi)*phi(zi)] = E[zi*phi(zi)] for h=0!!!
Ephizij=triu(Ephizij,1)+tril(Ephizij,-1)+diag(diag(Eziphizj,0));

% (c) 2017 Daniel Durstewitz, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
% adapted 2019 Georgia Koppe, Dept. Theoretical Neuroscience, Central
% Institute of Mental Health, Heidelberg University
