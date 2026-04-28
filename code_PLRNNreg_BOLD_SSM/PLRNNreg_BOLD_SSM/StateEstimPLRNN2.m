function [z,U,d,Err]=StateEstimPLRNN2(net,Inp_,X_,rp,z0,d0,tol,eps,flipAll)
% MODIFIED StateEstimPLRNN2.m 
% 1. Forces Column/Row consistency for v0, v1a, v1b construction.
% 2. Adds safety check for empty violations (prevents index error).
% 3. Includes optimization breaks (n < 50) and robust solver (pinv).

% Get parameters
A=net.A; W=net.W; S=net.Sigma; mu0_=net.mu0;
J=net.J; B=net.B; G=net.Gamma; C=net.C; h=net.h;
H=net.getConvolutionMtx;

if nargin<7, tol=1e-2; end
if nargin<8, eps=[]; end
if nargin<9, flipAll=false; end

m=length(A);   
if iscell(X_), X=X_; Inp=Inp_; mu0=mu0_; else, X{1}=X_; Inp{1}=Inp_; mu0{1}=mu0_; end
ntr=length(X);  

% Hessian blocks
u0=S^-1+A'*S^-1*A; K0=-A'*S^-1;
u2a=W'*S^-1*W; u2b=B'*G^-1*B;
u1=W'*S^-1*A; K2=-W'*S^-1;

% Pre-calculate total size to prevent dynamic growth issues
total_T = sum(cellfun(@(x) size(x,2), X));
total_dim = m * total_T;

U0=sparse(total_dim, total_dim); 
U2a=sparse(total_dim, total_dim); 
U2b=sparse(total_dim, total_dim); 
U1=sparse(total_dim, total_dim);

v0 = zeros(total_dim, 1);
v1a = zeros(total_dim, 1);
v1b = zeros(total_dim, 1);

current_idx = 0;

for i=1:ntr  
    T=size(X{i},2); 
    trial_dim = m*T;
    range = (current_idx + 1) : (current_idx + trial_dim);
    
    % --- 1. Build Sparse Matrices (Block Diagonal) ---
    U0_ = kron(speye(T), u0); KK0 = kron(speye(T), K0); 
    U2a_ = kron(speye(T), u2a); U2b_ = kron(speye(T), u2b); 
    U1_ = kron(speye(T), u1); KK2 = kron(speye(T), K2);
    
    KK0=blkdiag(KK0,K0);
    kk=(T-1)*m+1:T*m; U0_(kk,kk)=S^-1;
    U0_=U0_+KK0(m+1:end,1:T*m);     
    KK0=KK0'; U0_=U0_+KK0(1:T*m,m+1:end);
    U2a_(kk,kk)=zeros(m,m); U2b_(kk,kk)=B'*G^-1*B;                   
    U1_(kk,kk)=0; KK2=blkdiag(KK2,K2); U1_=U1_+KK2(m+1:end,1:T*m);
    
    U0(range, range) = U0_; U2a(range, range) = U2a_;
    U2b(range, range) = U2b_; U1(range, range) = U1_;
    
    % --- 2. Build Vectors (Robust Concatenation) ---
    I_inp = C*Inp{i} + repmat(h,1,T);
    
    % v0 construction
    vka = S^-1 * I_inp; 
    vka(:,1) = vka(:,1) + S^-1 * (mu0{i}-h);     
    vkb = A' * S^-1 * I_inp(:,2:T);
    
    % Force column vectors using (:)
    term1 = vka(:);
    term2 = [vkb(:); zeros(m,1)]; 
    v0(range) = term1 - term2;
    
    % v1a construction
    vkb_w = -W' * S^-1 * I_inp(:,2:T);
    v1a(range) = [vkb_w(:); zeros(m,1)];
    
    % v1b construction
    vka_b = B' * G^-1 * X{i};
    vr1 = B' * G^-1 * J * rp';
    v1b(range) = vka_b(:) - vr1(:);
    
    current_idx = current_idx + trial_dim;
end

% --- Initialization ---
n=1; idx=[]; k=[];
% Ensure d matches the total dimension perfectly
if nargin>4 && ~isempty(z0), z=z0(1:end)'; else, z=randn(total_dim,1); end    
if nargin>5 && ~isempty(d0), d=d0; else, d=zeros(1,total_dim); d(z>0)=1; end     

% Force d to be row vector
if size(d,1) > 1, d = d'; end

Err=1e16;
y=rand(total_dim,1); LL=d*y; 
U=[]; dErr=-1e8;

MAX_LOOPS = 50; % Optimization

% --- Iteration Loop ---
idx = find(abs(d-(z'>0))); % Initial check

while (n==1 || ~isempty(idx)) && isempty(k) && dErr<tol*Err(n) && n < MAX_LOOPS
    zsv=z; Usv=U; dsv=d;
    if n>1
        if flipAll, dsv(idx)=1-d(idx);
        else dsv(idx(r))=1-d(idx(r)); end
    end
    
    % Build Hessian
    D=spdiags(d',0,total_dim,total_dim);   
    A1=U0+D*U2a*D'+(D*U1)+(D*U1)';  
    A2=H'*U2b*H;                    
    U=A1+1/2*(A2+A2');
   
    if ~isempty(eps), U=U+eps*speye(size(U)); end   
    
    % Build RHS (Transpose v0, v1a to match d logic if needed)
    % v0 is Col, v1a is Col, d is Row. d'.*v1a is Col.
    % So B1 is Col.
    B1 = v0 + (d' .* v1a); 
    B2 = H' * v1b;         
    
    % Solve
    try
        z = U \ (B1+B2);
    catch
        z = pinv(full(U)) * (B1+B2);
    end

    % Check constraints
    % d is Row, z is Col. z'>0 is Row.
    diff_vec = d - (z' > 0);
    idx = find(abs(diff_vec));
    
    if isempty(idx)
        break; 
    end
    
    ae = abs(z(idx));
    
    n=n+1; 
    Err(n)=sum(ae); 
    if n > 1, dErr=Err(n)-Err(n-1); end
    
    if flipAll
        d(idx) = 1 - d(idx);                    
    else
        [~,r] = max(ae); 
        if r <= length(idx)
            target_idx = idx(r);
            d(target_idx) = 1 - d(target_idx); 
        end
    end  

    l=d*y; k=find(LL==l); LL=[LL l]; 
end

if dErr < tol*Err(n)
    z=reshape(z,m,total_dim/m);
    disp(['#1 - ' num2str(dErr) '   ' num2str(Err(end))])
else
    z=reshape(zsv,m,total_dim/m);
    U=Usv; d=dsv; 
    disp(['#2 - ' num2str(length(idx)) '   ' num2str(length(k))])
end