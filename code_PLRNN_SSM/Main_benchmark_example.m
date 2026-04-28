clear; close all; clc

%->set paths for state space models and data
pat_PLRNNmodel=[pwd '/PLRNN'];
pat_LDSmodel=[pwd '/LDS'];
% pat_data=[pwd '/data/Lorenz/'];
pat_data=[pwd '/data/VDP/'];
pat_output=[pat_data '/output/']; 
if ~isfolder(pat_output), mkdir(pat_output); end
path(path,pat_PLRNNmodel);
path(path,pat_LDSmodel)

%-> specify data file
str=['*T1000*.mat'];
files=dir([pat_data str]);
nfiles=size(files,1);

mstates=[8 10 12 14];    %set latent state dimension
%pp=parpool(48); %for running parallel files

lam=0;
for m=mstates
    M=m;
    %par 
    for i=1:nfiles
        datafilename=files(i).name;
        disp(datafilename)
        datafile=[pat_data datafilename];
        outputfile=[datafilename(1:end-4) '_M' num2str(M), '.mat'];
        
        %->load data
        dat=load(datafile);
        runAlgorithm1(M,lam,dat,pat_output,outputfile)

    end
end
%delete(gcp('nocreate'))
