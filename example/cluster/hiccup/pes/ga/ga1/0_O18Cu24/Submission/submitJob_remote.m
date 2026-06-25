function jobNumber = submitJob_remote(USPEX, Index)
%-------------------------------------------------------------
%This routine is to check if the submitted job is done or not
%2   : whichCluster (default 0, 1: local submission; 2: remote submission)
%C-20GPa : remoteFolder
%-------------------------------------------------------------

%-------------------------------------------------------------
%Step1: To prepare the job script, runvasp.sh
  fp = fopen('runvasp.sh', 'w');
  fprintf(fp, '#!/bin/sh\n');
  fprintf(fp, '#PBS -l nodes=2:ppn=2,walltime=1:30:00\n');
  fprintf(fp, '#PBS -N USPEX\n');
  fprintf(fp, '#PBS -j oe\n');
  fprintf(fp, '#PBS -V \n');
  fprintf(fp, 'cd ${PBS_O_WORKDIR}\n');
  fprintf(fp, '/usr/local/pkg/openmpi-1.4.5/bin/mpirun -np 4 vasp1 > vasp.out\n');
  fclose(fp);
%------------------------------------------------------------------------------------------------------------
%Step 2: Copy the files to the remote machine

%Step2-1: Specify the PATH to put your calculation folder
Home = ['/nfs/user08/qiazhu']; %'pwd' of your home directory of your remote machine
Address = 'qiazhu@seawulf.stonybrook.edu'; %your target server: username@address
Path = [Home '/' USPEX '/CalcFold' num2str(Index)];  %Just keep it

%Step2-2: Create the remote directory 
%Please change the ssh/scp command if necessary!!!!!!!!!!!!!!!!!!
try
[a,b]=unix(['ssh -i ~/.ssh/seawulf ' Address ' mkdir ' USPEX ]);  
catch
end

try
[a,b]=unix(['ssh -i ~/.ssh/seawulf ' Address ' mkdir ' Path ]);
catch
end

%Step2-3: Copy necessary files
[nothing, nothing] = unix(['scp -i ~/.ssh/seawulf POSCAR   ' Address ':' Path]);
[nothing, nothing] = unix(['scp -i ~/.ssh/seawulf INCAR    ' Address ':' Path]);
[nothing, nothing] = unix(['scp -i ~/.ssh/seawulf POTCAR   ' Address ':' Path]);
[nothing, nothing] = unix(['scp -i ~/.ssh/seawulf KPOINTS  ' Address ':' Path]);
[nothing, nothing] = unix(['scp -i ~/.ssh/seawulf runvasp.sh ' Address ':' Path]);

%------------------------------------------------------------------------------------------------------------
%Step 3: to submit the job and get JobID, i.e. the exact command to submit job.
[a,v]=unix(['ssh -i ~/.ssh/seawulf ' Address ' /usr/local/pkg/torque/bin/qsub ' Path '/runvasp.sh'])

% format: Job 1587349.nagling is submitted to default queue <mono>
end_marker = findstr(v,'.');
if strfind(v,'error')
   jobNumber=0;
else
   jobNumber = v(1:end_marker(1)-1);
end

