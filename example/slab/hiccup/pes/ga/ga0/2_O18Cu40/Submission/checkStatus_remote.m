function doneOr = checkStatus_remote(jobID, USPEX, Folder)
%--------------------------------------------------------------------
%This routine is to check if the submitted job is done or not
%One needs to do a little edit based on your own case.
%--------------------------------------------------------------------

%Step1: Specify the PATH to put your calculation folder
Home = ['/nfs/user08/qiazhu']; %'pwd' of your home directory of your remote machine
Address = 'qiazhu@seawulf.stonybrook.edu';  %Your target supercomputer: username@address.
Path = [Home '/' USPEX '/CalcFold' num2str(Folder)]; %just keep it

%Step2: Check JobID, the exact command to check job by jobID
[a,b]=unix(['ssh -i ~/.ssh/seawulf ' Address ' /usr/local/pkg/torque/bin/qstat ' num2str(jobID)])
    tempOr1 = strfind(b, 'R batch');
    tempOr2 = strfind(b, 'Q batch');
    if isempty(tempOr1) & isempty(tempOr2)
      doneOr = 1;
%     [nothing, nothing] = unix(['scp -i ~/.ssh/seawulf ' Address ':' Path '/OUTCAR ./']) %OUTCAR is not necessary by default
      [nothing, nothing] = unix(['scp -i ~/.ssh/seawulf ' Address ':' Path '/OSZICAR ./']) %For reading enthalpy/energy
      [nothing, nothing] = unix(['scp -i ~/.ssh/seawulf ' Address ':' Path '/CONTCAR ./']) %For reading structural info
    else
       doneOr = 0;
    end
