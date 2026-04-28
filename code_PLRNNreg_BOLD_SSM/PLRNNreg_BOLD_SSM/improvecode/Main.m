clear; close all
%addpath('/home/georgia.koppe/Documents/social neuroscience/GamblingFMRI/batchs/fMRI/spm12')

%ntest
% set parameters for class initialization
p = 6;          % hidden states
q = 10;         % observed states 
T = 15;        % time
N = T;        % change W every 15 time steps
TR = 3;         % scan repetition time
m = 3;          % number of nuiscance vars
k = p;          % dimension of input

%i want to map the first 5 obs. to the first 3 states...

% Create instance of a linear RNN
net0 = PLRNN_fMRI(p,q,T,TR,N,m,k);
seed=clock;
net0.init_random_parameters(seed(end));

%separate first 5 and second 5 obs (belonging to 2 networks)
%each is explained by 3 states
%Nx=5; Mz=3;
%net0.B(1:Nx,Mz+1:net0.p)=0;
%net0.B(Nx+1:net0.q,1:Mz)=0;
net0.B(1:2,3:end)=0;
net0.B(3:5,[1 2 6])=0;
net0.B(6:10,[1:5])=0


Bmap=net0.B~=0;%[ones(Nx,Mz), zeros(Nx,Mz); zeros(Nx,Mz) ones(Nx, Mz)];
net0.Bmap=Bmap;

% provide experimental input
k=p;
r=(rand(k,T)>0.7); Inp{1}(r)=3-2*1;
Inp=r+0;

% Run network
Z = net0.run_network(Inp);

%generate observations
%--------------------------------------------------------------------------
rp=1*randn(T,m);
th=repmat(net0.h,1,T);
H=net0.getConvolutionMtx(1);
hZ=H*Z';
X=net0.B*hZ'+net0.M*rp'+mvnrnd(zeros(1,q),net0.Gamma,T)';
%hh=repmat(net0.h,1,T);
hh=zeros(p,T);
hh=reshape(hh,1,T*p);
z0=reshape(Z,1,T*p); d0=zeros(1,p*T); d0(z0>0)=1;

figure; hold on
subplot(1,2,1);plot(X'); title('X')
subplot(1,2,2); plot(Z'); title('Z')

%--------------------------------------------------------------------------
net = copy(net0);
%net.init_random_parameters;

%EM
%--------------------------------------------------------------------------
%perform estimation and maximization

%E-step
tol = 1e-5;      % tolerance (EM)
maxiter = 100;   % max iterations (EM)
flipAll=false;
tic
[Ezi,U,~,~]=StateEstimPLRNN2(net,Inp,X,rp,z0,d0,tol,eps,flipAll);
toc

% tic
% [Ezi2,U2,~,~]=StateEstimPLRNN_sp(net,Inp,X,rp,z0,d0,tol,eps,flipAll);
% toc
% keyboard

figure; hold on;
subplot(1,3,1); hold on
plot(Z'); title('true Z');
subplot(1,3,2); hold on
plot(Ezi'); title('E[Z]')
subplot(1,3,3); hold on
plot((Ezi-Z)'); title('E[Z]- true Z')
tic
%[Ephizi,Ephizij,Eziphizj,Vest]=CompExpecPLRNN_sp(net,Z,U);
[Ephizi,Ephizij,Eziphizj,Vest]=ExpValPLRNN3(net,Z,U);

toc
tic
% M-step
fixedS=1; 
fixedG=0;
fixedB=0;
fixedC=0;
lam=0;
[net]=ParamEstimPLRNN_sp(net,Z,Vest,Ephizi,Ephizij,Eziphizj,X,Inp,rp,fixedS, fixedC, fixedG, fixedB);
toc


%perform EM with iterations
%MaxIter=35;
%EM
%[net,LL,Ezi,Ephizi,Ephizij,Eziphizj]=EMiter_sp(net,tol,eps,MaxIter,X,Inp,rp);
%--------------------------------------------------------------------------

%Quality check
%--------------------------------------------------------------------------
figure('color','white');
Pest={'net.mu0','net.B','net.Gamma','net.A','net.M','net.h','net.C'};
Test={'net0.mu0','net0.B','net0.Gamma','net0.A','net0.M','net0.h','net0.C'};
figure('color','white');
for i=1:length(Test)
    eval(['pred=' Pest{i} ';'])
    eval(['tru=' Test{i} ';'])
    tmp1=reshape(pred,1,numel(pred));
    tmp2=reshape(tru,1,numel(tru));
    xmin=min([tmp1,tmp2]); xmax=max([tmp1 tmp2]);
    subplot(3,3,i); hold on; title(Test{i}); box on
    plot(tmp1,tmp2,'ob','LineWidth',2); %lsline
    ss=(xmax-xmin)/20; plot(xmin:ss:xmax,xmin:ss:xmax,'-r');
    xlim([xmin xmax]); ylim([xmin xmax])
    xlabel('predicted'); ylabel('true');
    clear pred tru
end

figure('color','white');
for i=1:net.n
    wti=net0.W(:,:,i);
    wpi=net.W(:,:,i);
    tmp1=reshape(wti,1,numel(wti));
    tmp2=reshape(wpi,1,numel(wpi));
    % plot(tmp1,tmp2,'ob','LineWidth',2); %lsline
    xmin=min([tmp1,tmp2]);
    xmax=max([tmp1 tmp2]);
   % subplot(net.n/2,2,i); 
    hold on;
    title(['W' num2str(i)]); box on
    ss=(xmax-xmin)/20; plot(xmin:ss:xmax,xmin:ss:xmax,'-r');
    plot(tmp1,tmp2,'ob','LineWidth',2);
    xlim([xmin xmax])
    xlabel('predicted'); ylabel('true');
end
keyboard
delete(net); delete(net0);