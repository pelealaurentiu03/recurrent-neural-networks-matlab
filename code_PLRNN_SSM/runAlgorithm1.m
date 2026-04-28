function runAlgorithm1(M,lamda,dat,pat_output,outputfile)

XX=dat.X;
T=size(XX,2);
X{1}=XX(:,:,1);
Inp{1}=zeros(M,T);

[N,T]=size(X{1});
ntr=1;
lam= ntr*lamda;

%% initialization
%--------------------------------------------------------------------------
sd=1;
a=-1.5*sd; b=1.5*sd; 
W0=a+(b-a).*rand(M);
E=eig(W0);
redf=.95;
while find(abs(E)>=1), W0=W0*redf; E=eig(W0); end
%--------------------------------------------------------------------------
C=zeros(M);
h0=0.5*randn(M,1);
B0=randn(N,M);
for i=1:length(X), mu00{i}=randn(M,1); end


%% --- 1st step: LDS; B,G free to vary
%--------------------------------------------------------------------------
tol=1e-3;   % relative (to first LL value) tolerated increase in log-likelihood
MaxIter=10; % maximum number of EM iterations allowed
eps=1e-5;   % singularity parameter in StateEstPLRNN
fixedS=1;   % S to be considered fixed or to be estimated
fixedC=1;   % C to be considered fixed or to be estimated
fixedB=0;   % B to be considered fixed or to be estimated
fixedG=0;
CtrPar=[tol MaxIter eps fixedS fixedC fixedB fixedG];

S=eye(M);
G0=diag(var(cell2mat(X)'));  % take data var as initial estim (~1 here)
%  --> actually important to determine some scale of noise a priori!

[mu01,B1,G1,W1,h1]=EMiter_LDS2pari1(CtrPar,W0,C,S,Inp,mu00,B0,G0,h0,X);
disp('first iteration done')

%% --- 2nd step: estimate PLRNN model with B,G free to vary
%--------------------------------------------------------------------------

fixedB=0;   % B to be considered fixed or to be estimated
fixedG=0;   % G to be considered fixed or to be estimated
S=eye(M);   

A1=diag(diag(W1)); 
W1=W1-A1;

tol2=1e-2;      % relative error tolerance in state estimation (see StateEstPLRNN)
flipOnIt=10;    % parameter that controls switch from single (i<=flipOnIt) to all
FinOpt=0;       % quad. prog. step at end of E-iterations
CtrPar=[tol MaxIter tol2 eps flipOnIt FinOpt fixedS fixedC fixedB fixedG];

[mu02,B2,G2,A2,W2,~,h2]= EMiter3pari1(CtrPar,A1,W1,C,S,Inp,mu01,B1,G1,h1,X,[],[],[],lam);  
disp('second iteration done')

%% --- 3rd step: estimate PLRNN model with B fixed & S=.1
%--------------------------------------------------------------------------
S=0.1*eye(M);
fixedB=1;
CtrPar=[tol MaxIter tol2 eps flipOnIt FinOpt fixedS fixedC fixedB fixedG];

[mu03,B3,G3,A3,W3,~,h3]= EMiter3pari1(CtrPar,A2,W2,C,S,Inp,mu02,B2,G2,h2,X);
disp('third iteration done')

%% --- 4th step: estimate PLRNN model with B fixed & S=.01
%--------------------------------------------------------------------------
S=0.01*eye(M);
[mu04,B4,G4,A4,W4,~,h4]= EMiter3pari1(CtrPar,A3,W3,C,S,Inp,mu03,B3,G3,h3,X);
disp('fourth iteration done')

%% --- 5th step: estimate PLRNN model with B fixed & S=.001
%--------------------------------------------------------------------------
S=0.001*eye(M);
[mu0,B,G,A,W,~,h,~,Ezi,Vest,~,~,~,LL]= ...
    EMiter3pari1(CtrPar,A4,W4,C,S,Inp,mu04,B4,G4,h4,X);
disp('fifth iteration done')

%% --- save
save([pat_output outputfile ],'X','mu0','B','G','A','W','h','S','C','LL','Ezi', 'lam','Vest');



