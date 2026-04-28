%(c) 2019, Georgia Koppe, Dept. of Theoretical Neuroscience,
%Central Institute of Mental Health, Heidelberg University, Mannheim
% Modified by Andrei on Dec 18.12.2025) - Added Adaptive Mask, Cell Unwrap and Dim Fix
%--------------------------------------------------------------------------
%Nonlinear RNN class
%z_t=f(z_{t-1})=A*z_{t-1}+W*max(0,z_{t-1})+h+I_t+ e,  e~N(0,Sigma)  
%x_t=g(z_t)=B*h(z_t) + J*r + n,     n~N(0,Gamma)
% see model details in Koppe et al 2019, PLOS Computational Biology
%--------------------------------------------------------------------------
classdef PLRNN_BOLD < matlab.mixin.Copyable
    
    properties
        %general dimensionalities
        p           %dim latent states
        q           %dim observations
        k           %dim input
        T           %time steps
        TR          %scan repetition time
        hrf         %hemodynamic response function
        m           %number of nuiscance variables in observation
        randomseed  %log away random seed for initial conditions
        X           %observations
        R           %nuiscance covariates
        Inp         %inputs
        Z           %estimates of latent states
        
        %model parameters
        B           %observation: transform
        J           %observation: nuisance regression matrix 
        Gamma       %observation: noise
        
        A           %evolution: linear transfo
        W           %evolution: piecewise linear transform
        C           %evolution: input regression matrix
        h           %evolution: threshold theta
        Sigma       %evolution: noise
        mu0         %evolution: initial latent state
        
        Bmap        %constraint matrix defining which elements in B are set to 0
        reg         %first order regularization terms  
    end
    
    % Class functions/methods
    methods
        
        % Constructor
        %------------------------------------------------------------------
        function obj = PLRNN_BOLD(p,q,T,TR,m,X,R,U,k)
            obj.p = p;   %number of latent dimensions
            obj.q = q;   %number of observations
            obj.T = T;   %number of time steps
            obj.TR= TR;  %repetition time for fMRI
            obj.m = m;   %number of nuisance variables (e.g. movement)
            obj.hrf=obj.getHRF;
            obj.X=X;     %observations
            obj.R=R;     %nuiscance covariates
            obj.Inp=U;
            obj.k = k;   %dimension of input
        end
        
        % get hemodynamic response function
        % uses spm_hrf.m (Copyright (C) 1996-2015 Wellcome Trust Centre for Neuroimaging; Karl Friston)
        %------------------------------------------------------------------
        function hrf=getHRF(obj)
            hrf=spm_hrf(obj.TR);
        end        
       
        % initializes parameters
        %------------------------------------------------------------------
        function init_pars(obj,seed)
            
            if nargin<2,s=ceil(abs(10*randn)); else s=seed; end
            obj.randomseed=s;
            rand('state',s); randn('state',s);
            %latent process parameters
            W0=2*randn(obj.p, obj.p);
            E=eig(W0);
            redf=.95;
            while find(abs(E)>=1), W0=W0*redf; E=eig(W0); end
            obj.A=diag(diag(W0));
            obj.W=W0-obj.A;
            obj.h=.5*randn(obj.p,1);
            obj.C=zeros(obj.p, obj.k);
            obj.Sigma=eye(obj.p);   
            obj.mu0=randn(obj.p,1);
            
            %observation parameters
            obj.B=randn(obj.q,obj.p);
            obj.Gamma=diag(var(obj.X'));  
            
            %initialize J by GLM
            XX=[ones(obj.T,1) obj.R]; 
            Y=obj.X';
            obj.J=(XX'*XX)^-1*XX'*Y;
            obj.J=obj.J(2:end,:);  
            obj.J=obj.J';
        end
        
        % evolution function f with noise
        %------------------------------------------------------------------
        function x_next = f( obj, x, I,trial)
            err=mvnrnd(zeros(obj.p,1),obj.Sigma)';
            x_next = obj.A*x + obj.W(:,:,trial)*max(x,0) +obj.h+ obj.C*I + err;
        end
        
        % observation function with noise
        %------------------------------------------------------------------
        function y_next = g(obj, x)
            err=mvnrnd(zeros(obj.q,1),obj.Gamma)';
            H=obj.getConvolutionMtx;
            
            y_next = obj.B * H*x + err;
        end
        
        % get convolution matrix 
        %------------------------------------------------------------------
        function H=getConvolutionMtx(obj,dim)
            if nargin<2, L=obj.p; else L=dim; end
            
            l=1;
            for i=1:length(obj.hrf)
                zhrf(l)=obj.hrf(i); l=l+1;
                for j=2:L
                    zhrf(l)=0; l=l+1;
                end
            end
            Hconv=convmtx(zhrf',L*obj.T);
            H=Hconv(1:obj.T*L,:);
        end
        
        %get diagonal hrf structure needed for parameter estimation
        %---------------------------------------------------------------------------
        function hrft=getDiagHrf(obj,t)
            l=obj.hrf';
            if t<length(l)
                hrft=diag(fliplr(l(1:t)));
            else
                hrft=diag(fliplr(l));
            end
        end
        %returns trial for W referencing t in z_t (sum t=2:T) 
        %note: not implemented yet, unnecessary use...
        %------------------------------------------------------------------
        function [trial ]= get_trial_curr( obj,t )
            trial=1;
        end
        
        % run network
        %------------------------------------------------------------------
        function [x, y] = run_network(obj, I)
            
            x = zeros(obj.p, obj.T);
            y = zeros(obj.q, obj.T);
            x(:,1)=obj.mu0+obj.C*I(:,1);
   
            for t=1:obj.T-1
                trial=1;
                x(:,t+1) = f(obj, x(:,t), I(:,t+1),trial);
            end
            
        end
        
        % log likelihood (Modified for Robustness)
        %------------------------------------------------------------------
        function LL = ComplLogLike(obj,Inp,X,Z,rp,fixedC)
            
            % --- FIX 1: Unwrap Cell Arrays ---
            if iscell(Inp), Inp = Inp{1}; end
            if iscell(X), X = X{1}; end
            
            % --- FIX 2: Ensure Input Dimensions Match Time (T) ---
            % Inp should be [K x T]. If it is [T x K], we transpose it.
            % We anchor this check on obj.T (number of time steps).
            if size(Inp, 2) ~= obj.T && size(Inp, 1) == obj.T
                Inp = Inp';
            end
            % -----------------------------------------------------

            % add regularization
            reg=obj.reg;
            if ~isempty(reg)
                tau=reg.tau;
                lambda=reg.lambda;
                
                if fixedC
                    L=[obj.A obj.W obj.h];
                else
                    L=[obj.A obj.W obj.h obj.C];
                end
                LMask=reg.Lreg;
                
                % --- FIX 3: Adaptive Mask Sizing ---
                % Ensures LMask matches the dimensions of L before element-wise op
                if ~isequal(size(L), size(LMask))
                    new_mask = zeros(size(L));
                    r = min(size(L,1), size(LMask,1));
                    c = min(size(L,2), size(LMask,2));
                    new_mask(1:r, 1:c) = LMask(1:r, 1:c);
                    LMask = new_mask;
                end
                % -----------------------------------
                
                %regularization on pars->0
                Lhat=L.*(LMask==1);
                Reg1=-(tau/2)*trace(Lhat'*Lhat);
                
                %regularization on pars->1
                Lhat=L.*(LMask==-1);
                OneMtx=(LMask==-1).*(L~=0);
                Reg2=-(lambda/2)*trace((Lhat-OneMtx)'*(Lhat-OneMtx));
            else
                Reg1=0; Reg2=0;
            end
                                       
            % complete log-likelihood
            Hrf=obj.getConvolutionMtx(1);
            
            % --- C Reshape (Just in case) ---
            if size(obj.C, 2) ~= size(Inp, 1)
                 % If C was initialized with different dim than Inp, fix C
                 obj.C = zeros(obj.p, size(Inp, 1));
            end
            % -------------------------------------------------
            
            P(:,1)=Z(:,1)-obj.mu0-obj.C*Inp(:,1);
            for t=2:obj.T
                trial=1;
                P(:,t)= Z(:,t)-obj.A*Z(:,t-1)-obj.W(:,:,trial)*max(0,Z(:,t-1))-obj.h-obj.C*Inp(:,t);
            end
            LL1=trace(P'*obj.Sigma^-1*P);
            hZ=Z*Hrf';                          
            P=X-obj.B*hZ-obj.J*rp';
            LL2=trace(P'*obj.Gamma^-1*P);
            LL=-1/2*(LL1+LL2+obj.T*sum(log(diag(obj.Gamma)))+obj.T*log(det(obj.Sigma)))+Reg1+Reg2;
        end
        
    end
end