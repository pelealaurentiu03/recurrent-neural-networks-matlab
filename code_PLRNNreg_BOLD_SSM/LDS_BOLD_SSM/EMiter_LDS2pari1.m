function [net,LL,Ezi,Vest]=EMiter_LDS2pari1(net,CtrPar,X,Inp,rp)

i=1; LLR=1e8; LL=[];
Ezi=[];

tol=CtrPar(1);
MaxIter=CtrPar(2);
eps=CtrPar(3);
fixedS=CtrPar(4);   % S to be considered fixed or to be estimated
fixedC=CtrPar(5);   % C to be considered fixed or to be estimated
fixedB=CtrPar(6);   % B to be considered fixed or to be estimated
fixedG=CtrPar(7);   % G to be considered fixed or to be estimated

while (i==1 || LLR>tol*abs(LL(1)) || LLR<0) && i<MaxIter
    
    % E-step---------------------------------------------------------------
    [Ezi,Vest]=StateEstimLDS_sp(net,Inp,X,rp,eps);
        
    % M-step---------------------------------------------------------------
    net=ParamEstimLDS_sp(net,Ezi,Vest,X,Inp,rp, fixedS, fixedC, fixedG, fixedB);

    % complete log-likelihood
    LL(i)=ComplLogLikelihood(net,Inp,X,Ezi,rp);
    disp(LL(i))

    if i>1, LLR=LL(i)-LL(i-1); else LLR=1e8; end
    i=i+1;
end
disp(['LL, # iterations = ' num2str(LL(end)) ', ' num2str(i-1)]);


% (c) 2017 Daniel Durstewitz, Dept. Theoretical Neuroscience,
% adapted: 2018 Georgia Koppe, Dept. Theoretical Neuroscience
% Central Institute of Mental Health Mannheim, Heidelberg University
