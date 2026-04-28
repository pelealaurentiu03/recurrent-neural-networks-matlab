%(c) Georgia Koppe, 2019
% example for starting PLRNN-BOLD-SSM estimation with Algorithm-1
% 04/11/2019 updated with new regularization possibilities
%--------------------------------------------------------------------------

clear; close all; clc

%->set/add relevant directories
startpath=pwd;
patPLRNN=[startpath '/PLRNNreg_BOLD_SSM'];     %PLRNN-BOLD-SSM model directory
patLDS=[startpath '/LDS_BOLD_SSM'];         %LDS-BOLD-SSM model directory

patData=[startpath '/data/'];               %data directory
patOut=[patData 'output/'];                 %output directory
if ~isfolder(patOut), mkdir(patOut); end
addpath(patPLRNN);

%->set options
opt=2;      %case 1=without inputs, 2=with inputs
lam=50;     %regularization parameter
M=3;        %latent state dim


%->find all data files
str='*.mat';
files=dir([patData str]);
nfiles=size(files,1);

k=1;    %do protocol for first file
filename=files(k).name; %one example file

%->load data
load([patData filename])
[~,name]=fileparts(filename);
txtvpn=name(end-2:end);

X=PLRNN.data;   %X / data
R=PLRNN.rp;     %R / nuiscance effects
Inp=PLRNN.Inp;  %S / stimulus inputs
K=size(Inp,1);  %dimension of input
T=size(X,2);    %length of time series
N=size(X,1);    %dimension of output
P=size(R,2);    %dimension of nuiscance covariates
TR=PLRNN.preprocess.RT; %time of repetition

if opt==1
    disp('caution: inputs set to 0!');
    Inp=zeros(K,T);
end

%--------------------------------------------------------------------------
%---------------------------- MAIN (Algorithm-1)--------------------------%
%--------------------------------------------------------------------------

n=ceil(100*randn);
net=PLRNN_BOLD(M,N,T,TR,P,X,R,Inp,K); %create network instance
net.init_pars(n);  % initialize network randomly (step 0)

%  --> actually important to determine some scale of noise a priori!
tol=1e-3;       % relative tolerated increase in log-likelihood
MaxIter=20;    % maximum number of EM iterations allowed

% --- 1st step: estimate LDS model
%--------------------------------------------------------------------------

eps=1e-5;       % singularity parameter
fixedS=1;       % S to be considered fixed or to be estimated
fixedB=0;       % B to be considered fixed or to be estimated
fixedG=0;
fixedC=0;       % if external inputs, do not fix C
if opt==1, fixedC=1; end

CtrPar=[tol MaxIter eps fixedS fixedC fixedB fixedG];

S=eye(M);       %fix Sigma to certain value
net.Sigma=S;

addpath(patLDS) %add LDS to path
net=EMiter_LDS2pari1(net,CtrPar,X,Inp,R);
rmpath(patLDS)  %remove LDS from path
disp('first iteration done')


% --- 2nd step: estimate PLRNN model with B,G free to vary, S=1
%--------------------------------------------------------------------------

%add regularization
%--------------------------------------------------------------------------
% specify regularization
%A and/or W->1 regulated by lambda here set tau, A and/or W->0 regulated by tau
if fixedC
    LMask=zeros(size([net.A net.W net.h]));
else
    LMask=zeros(size([net.A net.W net.h net.C]));
end
%pars->1
Aind=1:ceil(M/2);    %regularize half of the states with A->1;
LMask(Aind,1:M)=-1;

%pars->0
Wind=1:ceil(M/2);       %which half of the states with W->0;
LMask(Wind,M+1:2*M)=1;
hind=1:ceil(M/2);
LMask(hind,2*M+1)=1;

tau=100;
%no regularization on C so far
reg.Lreg=LMask;
reg.lambda=tau; %specifies strength of regularization on pars->1
reg.tau=tau;   %specifies strength of regularization on pars->0
net.reg=reg;
%--------------------------------------------------------------------------

fixedB=0;       % B to be considered fixed or to be estimated
fixedG=0;       % G to be considered fixed or to be estimated
net.Sigma=eye(M);

tol2=1e-3;      % relative error tolerance in state estimation (see StateEstPLRNN)
flipAll=false;  % flip all bits at once?
CtrPar=[tol MaxIter tol2 eps flipAll fixedS fixedC fixedB fixedG];

net=EMiter3pari1(net,CtrPar,X,Inp,R);
disp('second iteration done')


% --- 3rd step: estimate PLRNN model with B & G fixed, S=.1
%--------------------------------------------------------------------------
S=0.1*eye(M);
net.Sigma=S;
fixedB=1;
fixedG=1;

CtrPar=[tol MaxIter tol2 eps flipAll fixedS fixedC fixedB fixedG];

net=EMiter3pari1(net,CtrPar,X,Inp,R);
disp('third iteration done')

% --- 4th step: estimate PLRNN model with B, G fixed & S=.01
%--------------------------------------------------------------------------
S=0.01*eye(M);
net.Sigma=S;

net=EMiter3pari1(net,CtrPar,X,Inp,R);
disp('fourth iteration done')

% --- 5th step: estimate PLRNN model with B, G fixed & S=.001
%--------------------------------------------------------------------------
S=0.001*eye(M);
net.Sigma=S;

[net,LL,Ezi,Ephizi,Ephizij,Eziphizj,V]=EMiter3pari1(net,CtrPar,X,Inp,R);
disp('fifth iteration done')


fileOut=[patOut 'outputfile' num2str(k) '.mat'];
save(fileOut,'net','LL','Ezi','Ephizi','Ephizij','Eziphizj','V');

%END
