function [VH]=getConvolvedCovanceMtx(Ezizi,net,Ezizij0g)

%get sub-covariance mtx for one state with the others
istate=1;
M=net.p;
T=net.T;

ind=istate:M:T*M;
V=full(Ezizi);
%convolve this one with hrf successively


m=M;
hrf=flipud(net.getHRF);
dim=length(hrf);
for i=1:dim
    hrfp(i,:)=[hrf(i) zeros(1,m-1)];
end

hE1=zeros(m,m); hhE2=zeros(m,m); hhE3=zeros(m,m);
for t=13:T
    tt=t-1; %caution, starts one earlier
    Cd=net.getDiagHrf(tt);
    Cdt=fliplr(Cd);
    tic
    dim=size(Cd,2);%-1;
    for im=1:m  %diagonal hrf components separately for each state
        qi=im:m:T*m; qj=im:m:T*m;
        Ephizijm=Ezizij0g(qi,qj);
        Ephizijtt=Ephizijm(tt-dim+1:tt,tt-dim+1:tt);
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
        Ephizijtt=Ephizijm(tt-dim+1:tt,tt-dim+1:tt);
        hhE(qi,qj)=sum(sum(Cd*Ephizijtt*Cdt));
    end
    hE1=hE1+hhE;
    toc
    disp('end time 1')
    %%%try to do step one in one go
    tic
    Ephizijtt=Ezizij0g(tt*m-dim*m+1:tt*m,tt*m-dim*m+1:tt*m); %sollten letzten dt schritte sein mit allen states statt nur einem (einer zu wenig?)
    hrf=reshape(hrfp',1,numel(hrfp));
    for i=1:m
        for j=i:m
            h=circshift(hrf,i-1);
            Hl=diag(h); Hr=fliplr(circshift(Hl,j-i));
            hhE2(i,j)=sum(sum(Hl*Ephizijtt*Hr));
            hhE2(j,i)=hhE2(i,j);
        end
    end
    toc
    disp('end time 2')
    
    %version 2
    tic
    HV=zeros(m*dim,m*dim);
    for i=1:m
        for j=i:m
            h=circshift(hrf,i-1);
            Hl=diag(h); Hr=fliplr(circshift(Hl,j-i));
            HV=HV+Hl*Ephizijtt*Hr;
        end
    end
%     for i=1:m
%             qi=i:m:m*dim;
%             qj=i:m:m*dim;
%             hhE3(i,i)=sum(sum(HV(qi,:)));
%     end
    disp('end time 3')
    
    toc
    
    keyboard
    
    
    hhE2
    hhE
    hhE-hhE2
    hhE-hhE3
    
    
    clear hhE hhE2
end
