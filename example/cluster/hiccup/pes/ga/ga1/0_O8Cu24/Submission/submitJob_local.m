function jobNumber = submitJob_local()
%-------------------------------------------------------------
%This routine is to check if the submitted job is done or not
%One needs to do a little edit based on your own case.

%1   : whichCluster (default 0, 1: local submission, 2: remote submission)
%-------------------------------------------------------------

%Step 1: to prepare the job script which is required by your supercomputer
fp = fopen('myrun', 'w');    
fprintf(fp, '#!/bin/sh\n');
fprintf(fp, '#PBS -l nodes=1:ppn=8,walltime=1:30:00 -q cfn_short\n');
fprintf(fp, '#PBS -N USPEX\n');
fprintf(fp, '#PBS -j oe\n');
fprintf(fp, '#PBS -V \n');
fprintf(fp, 'cd ${PBS_O_WORKDIR}\n');
fprintf(fp, 'mpirun -np 4 vasp1 > vasp.out\n');
fclose(fp);

%Step 2: to submit the job with the command like qsub, bsub, llsubmit, .etc.
%It will output some message on the screen like '2350873.nano.cfn.bnl.local'
[a,b]=unix(['qsub myrun'])


%Step 3: to get the jobID from the screen message
end_marker = findstr(b,'.');
jobNumber = b(1:end_marker(1)-1);   
