function LL=ComplLogLikelihood(obj,Inp,X,Z,rp)
            
% complete log-likelihood
Hrf=obj.getConvolutionMtx(1);
P(:,1)=Z(:,1)-obj.mu0-obj.C*Inp(:,1);
for t=2:obj.T
    trial=obj.get_trial_curr(t);
    P(:,t)= Z(:,t)-obj.A*Z(:,t-1)-obj.W(:,:,trial)*Z(:,t-1)-obj.h-obj.C*Inp(:,t);
end
LL1=trace(P'*obj.Sigma^-1*P);
hZ=Z*Hrf';                          
P=X-obj.B*hZ-obj.J*rp';
LL2=trace(P'*obj.Gamma^-1*P);
LL=-1/2*(LL1+LL2+obj.T*sum(log(diag(obj.Gamma)))+obj.T*log(det(obj.Sigma)));

