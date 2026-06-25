function jobNumber = submitJob(Ind_No)

global POP_STRUC
global ORG_STRUC

Step = POP_STRUC.POPULATION(Ind_No).Step;
code = ORG_STRUC.abinitioCode(Step);
dimension = ORG_STRUC.dimension;
jobNumber = 0;

if ORG_STRUC.abinitioCode(Step) == 0   % no optimization at all! (used in order optimization)
    jobNumber = 0.02;
    
elseif ORG_STRUC.platform == 0 %nonParallel
    
    jobNumber=100;
    [a,b]=unix(ORG_STRUC.commandExecutable{Step});
    if ORG_STRUC.numParallelCalcs > 1
        disp([ 'Individual ' num2str(Ind_No) ' @ step ' num2str(Step) ' is submitted']);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform == 1 %from USER local submission
    jobNumber = submitJob_local();
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform == 2 %from USER remote submission
    jobNumber = submitJob_remote(ORG_STRUC.remoteFolder, POP_STRUC.POPULATION(Ind_No).Folder);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform == 3 %CFN
    fp = fopen('myrun_CFN', 'w');
    fprintf(fp, '#!/bin/sh\n');
    fprintf(fp, '#PBS -l nodes=1:ppn=8,walltime=1:30:00\n');
    fprintf(fp, '#PBS -N USPEX\n');
    fprintf(fp, '#PBS -j oe\n');
    fprintf(fp, '#PBS -V \n');
    fprintf(fp, 'cd ${PBS_O_WORKDIR}\n');
    fprintf(fp, '/home1/qzhu/source/MPI/bin/mpirun -np 8 /home1/qzhu/source/vasp.5.2/vasp > vasp.out\n');
    fclose(fp);
    [a,b]=unix(['qsub myrun_CFN'])
    end_marker = findstr(b,'.');
    jobNumber = b(1:end_marker(1)-1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform==4 %QSH
    
    if code == 12
        numProc = 1;
        minProc = 1;
    else
        numProc = ORG_STRUC.numProcessors(Step);
        if numProc > 8
            minProc = 8;
            numProc = 8;
        else
            minProc = numProc;
        end
    end
    
    totalNowPath = [ORG_STRUC.homePath '/CalcFold' num2str(POP_STRUC.POPULATION(Ind_No).Folder)];
    [a,b]=unix(['cat /dev/null > myrun_QSH']);
    [a,b]=unix(['echo "#!/bin/sh"           >> myrun_QSH']);
    [a,b]=unix(['echo "#BSUB -sp 60"        >> myrun_QSH']);
    [a,b]=unix(['echo "#BSUB -q  intel"     >> myrun_QSH']);
    [a,b]=unix(['echo "#BSUB -a  intelmpi"  >> myrun_QSH']);
    [a,b]=unix(['echo "#BSUB -o  output"    >> myrun_QSH']);
    [a,b]=unix(['echo "#BSUB -R  \"span[ptile=' num2str(numProc) ']\"" >> myrun_QSH']);
    [a,b]=unix(['echo "#BSUB -J USPEX-',num2str(Ind_No),'S',num2str(POP_STRUC.POPULATION(Ind_No).Step),'"  >> myrun_QSH']);
    [a,b]=unix(['echo "#BSUB -W 06:00"      >> myrun_QSH']);
    [a,b]=unix(['echo "#BSUB -n ' num2str(minProc) ' "         >> myrun_QSH']);
    if code == 1
        [a,b]=unix(['echo "mpirun.lsf vasp-vdw"  >> myrun_QSH']);
    elseif code == 3
        [a,b]=unix(['echo "gulp < input > output"  >> myrun_QSH']);
    elseif code == 4
        [a,b]=unix(['echo "mpirun.lsf lammps < lammps.in "  >> myrun_QSH']);
    elseif code == 9
        [a,b]=unix(['echo "mpirun.lsf aims.x > FHI_output"  >> myrun_QSH']);
    elseif code == 11
        [a,b]=unix(['echo "RunCASTEP.sh -np 8 cstp >> log"  >> myrun_QSH']);
    elseif code == 12
        if dimension ~= -4
            [a,b]=unix(['echo "bash ./tinker.sh"  >> myrun_QSH']);
        else
            [a,b]=unix(['echo "python \$USPEXPATH/FunctionFolder/USPEX/M400/random_protein/random_protein.py input 1 pseudo memory > output"  >> myrun_QSH']);
        end
    elseif code == 13
        [a,b]=unix(['echo "MOPAC2012.exe input.mop"  >> myrun_QSH']);
    elseif code == 14
        [a,b]=unix(['echo "mpirun.lsf BoltzTraP BoltzTraP.def > BoltzTraP.output" >> myrun_QSH']);
    end
    [a,b]=unix(['bsub < myrun_QSH > job.info']);
    [a,b] =unix([' cat job.info | grep Job']);
    disp(b);
    start_marker=findstr(b,'<');
    end_marker = findstr(b,'>');
    jobNumber = b(start_marker(1)+1:end_marker(1)-1);
    disp([ 'Individual : ' num2str(Ind_No) ' -- JobID :', num2str(jobNumber) ]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform==5 %QSH2
    
    if code == 12
        numProc = 1;
        minProc = 1;
    else
        numProc = ORG_STRUC.numProcessors(Step);
        if numProc > 8
            minProc = 8;
            numProc = 8;
        else
            minProc = numProc;
        end
    end
    
    totalNowPath = [ORG_STRUC.homePath '/CalcFold' num2str(POP_STRUC.POPULATION(Ind_No).Folder)];
    [nothing, nothing] = unix(['cat /dev/null > myrun_QSH']);
    [nothing, nothing] = unix(['echo "#!/bin/sh"           >> myrun_QSH']);
    [nothing, nothing] = unix(['echo "#BSUB -sp 60"        >> myrun_QSH']);
    [nothing, nothing] = unix(['echo "#BSUB -q  amd"       >> myrun_QSH']);
    [nothing, nothing] = unix(['echo "#BSUB -a  intelmpi"  >> myrun_QSH']);
    [nothing, nothing] = unix(['echo "#BSUB -o  output"    >> myrun_QSH']);
    [nothing, nothing] = unix(['echo "#BSUB -R  \"span[ptile=' num2str(numProc) ']\"" >> myrun_QSH']);
    [nothing, nothing] = unix(['echo "#BSUB -J USPEX-',num2str(Ind_No),'S',num2str(POP_STRUC.POPULATION(Ind_No).Step),'"  >> myrun_QSH']);
    [nothing, nothing] = unix(['echo "#BSUB -W 06:00"      >> myrun_QSH']);
    [nothing, nothing] = unix(['echo "#BSUB -n ' num2str(minProc) ' "         >> myrun_QSH']);
    if code == 1
        [nothing, nothing] = unix(['echo "mpirun.lsf vasp-vdw"  >> myrun_QSH']);
    elseif code == 3
        [nothing, nothing] = unix(['echo "gulp < input > output"  >> myrun_QSH']);
    elseif code == 4
        [nothing, nothing] = unix(['echo "mpirun.lsf lammps < lammps.in > lammps.out "  >> myrun_QSH']);
    elseif code == 9
        [nothing, nothing] = unix(['echo "mpirun.lsf aims.x > FHI_output"  >> myrun_QSH']);
    elseif code == 11
        [a,b]=unix(['echo "RunCASTEP.sh -np 8 cstp >> log"  >> myrun_QSH']);
    elseif code == 12
        if dimension ~= -4
            [a,b]=unix(['echo "bash ./tinker.sh"  >> myrun_QSH']);
        else
            [a,b]=unix(['echo "python \$USPEXPATH/FunctionFolder/USPEX/M400/random_protein/random_protein.py input 1 pseudo memory > output"  >> myrun_QSH']);
        end
    elseif code == 13
        [a,b]=unix(['echo "MOPAC2012.exe input.mop"  >> myrun_QSH']);
    elseif code == 14
        [a,b]=unix(['echo "mpirun.lsf BoltzTraP BoltzTraP.def > BoltzTraP.output" >> myrun_QSH']);
    end
    [nothing, nothing] = unix(['bsub < myrun_QSH > job.info']);
    [a,b] =unix([' cat job.info | grep Job']);
    disp(b);
    start_marker=findstr(b,'<');
    end_marker = findstr(b,'>');
    jobNumber = b(start_marker(1)+1:end_marker(1)-1);
    disp([ 'Individual : ' num2str(Ind_No) ' -- JobID :', num2str(jobNumber) ]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform ==6 %xservDE
    
    [nothing, nothing] = unix('cat /dev/null > job');
    [nothing, nothing] = unix('echo "#\$ -S /bin/bash" > job');
    [nothing, nothing] = unix('echo "#\$ -N r30" >> job');
    [nothing, nothing] = unix('echo "#\$ -cwd" >> job');
    [nothing, nothing] = unix('echo "#\$ -l arch=darwin-x86" >> job');
    [nothing, nothing] = unix(['echo "#\$ -pe openmpi ' num2str(numProcessors) '" >> job']);
    [nothing, nothing] = unix('echo ". ~/.bashrc" >> job');
    [nothing, nothing] = unix('echo "hostname" >> job');
    [nothing, nothing] = unix(['echo "' ORG_STRUC.commandExecutable{Step} '" >> job']);
    % qsub script | awk '{print $3}'
    [a,v] = unix('qsub job');
    start_marker = 1; % format: Your job 248264 ("looki") has been submitted
    end_marker = findstr(v,')');
    jobNumber = v(10:end_marker(1)-2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform ==7 %MIPT-cluster
    totalNowPath = [ORG_STRUC.homePath '/CalcFold' num2str(POP_STRUC.POPULATION(Ind_No).Folder)];
    
    [nothing, nothing] = unix(['cat /dev/null > myrun_MIPT']);
    [nothing, nothing] = unix(['echo "#!/bin/sh"  >> myrun_MIPT']);
    [nothing, nothing] = unix(['echo "#SBATCH -o out"  >> myrun_MIPT']);
    [nothing, nothing] = unix(['echo "#SBATCH -p work "  >> myrun_MIPT']);
    [nothing, nothing] = unix(['echo "#SBATCH -J USPEX-' num2str(POP_STRUC.generation),'I',num2str(Ind_No),'S',num2str(POP_STRUC.POPULATION(Ind_No).Step),'"  >> myrun_MIPT']);
    [nothing, nothing] = unix(['echo "#SBATCH -t 06:00:00"  >> myrun_MIPT']);
    [nothing, nothing] = unix(['echo "#SBATCH -N 1"  >> myrun_MIPT']);
    
    [nothing, nothing] = unix(['echo cd ' totalNowPath ' >> myrun_MIPT']);
    [nothing, nothing] = unix(['echo mpdboot >>myrun_MIPT']);
    [nothing, nothing] = unix(['echo "mpirun -np 16  vasp-vdw> log" >> myrun_MIPT']);
    [nothing, nothing] = unix(['echo mpdallexit >>myrun_MIPT']);
    
    Ind_No
    [a,b] = unix(['sbatch myrun_MIPT '] );
    start_marker=findstr(b,'job ');
    jobNumber = b(start_marker(1)+4:end-1);
    disp([ 'Individual : ' num2str(Ind_No) ' -- JobID :', num2str(jobNumber) ]);
elseif ORG_STRUC.platform ==8 %NPU
    totalNowPath = [ORG_STRUC.homePath '/CalcFold' num2str(POP_STRUC.POPULATION(Ind_No).Folder)];
    cd (totalNowPath)
    %Step 1: to prepare the job script which is required by your supercomputer
    fp = fopen('jobrun', 'w');
    fprintf(fp, '#!/bin/sh \n');
    fprintf(fp, '#PBS -N USPEX \n');
    fprintf(fp, '#PBS -l nodes=1:ppn=8 \n');
    fprintf(fp, '#PBS -l walltime=6:00:00 \n');
    fprintf(fp, '#PBS -q batch \n');
    fprintf(fp, '#PBS -V \n');
    fprintf(fp, '#PBS -S /bin/bash \n');
    fprintf(fp, 'source /export/opensource/vasp.5.2/vaspenvSet.sh \n');
    fprintf(fp, 'EXEC=~/bin/vasp53 \n');
    fprintf(fp, 'MPI_HOME=/export/compiler/intel/impi/4.1.0.024 \n');
    fprintf(fp, 'NP=`wc -l < $PBS_NODEFILE` \n');
    fprintf(fp, ['cd ' totalNowPath ' \n']);
    fprintf(fp, '$MPI_HOME/bin/mpirun -machinefile $PBS_NODEFILE -np $NP $EXEC > out \n');
    fclose(fp);
    
    [a,b]=unix(['qsub jobrun']);
    end_marker = findstr(b,'.');
    jobNumber = b(1:end_marker(1)-1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform == 9 % CFN workshop 2014
    fp = fopen('myrun_CFN', 'w');
    fprintf(fp, '#!/bin/sh\n');
    fprintf(fp, '#PBS -l nodes=1:ppn=8,walltime=1:30:00 -q cfn_workshop\n');
    fprintf(fp, ['#PBS -N USPEX' num2str(Ind_No), 'S' ,num2str(POP_STRUC.POPULATION(Ind_No).Step) '\n']);
    fprintf(fp, '#PBS -j oe\n');
    fprintf(fp, '#PBS -V \n');
    fprintf(fp, 'ulimit -s unlimited\n');
    fprintf(fp, 'source /opt/intel/2013/bin/compilervars.sh intel64\n');
    fprintf(fp, 'cd ${PBS_O_WORKDIR}\n');
    fprintf(fp, '/software/mpi/openmpi/1.8.1-intel/bin/mpirun -np 8 /software/Workshop14/bin/vaspP > vasp.out\n');
    fclose(fp);
    [a,b]=unix(['qsub myrun_CFN'])
    end_marker = findstr(b,'.');
    jobNumber = b(1:end_marker(1)-1);
elseif ORG_STRUC.platform == 10 % UNN supercomputer (supz)
    totalNowPath = [ORG_STRUC.homePath '/CalcFold' num2str(POP_STRUC.POPULATION(Ind_No).Folder)];
    jobname = [num2str(POP_STRUC.generation),'I',num2str(Ind_No),'S',num2str(POP_STRUC.POPULATION(Ind_No).Step)];
    unix(['cat /dev/null > myrun_UNN']);
    unix(['echo "#!/bin/sh"  >> myrun_UNN']);
    unix(['echo "#SBATCH -o out"  >> myrun_UNN']);
    unix(['echo "#SBATCH -p all "  >> myrun_UNN']);
    unix(['echo "#SBATCH -J ' jobname '-USPEX"  >> myrun_UNN']);
    unix(['echo "#SBATCH -t 06:00:00"  >> myrun_UNN']);
    unix(['echo "#SBATCH -N 1"  >> myrun_UNN']);
    unix(['echo "#SBATCH -n 16"  >> myrun_UNN']);
    unix(['echo "" >> myrun_UNN']);
    unix(['echo cd ' totalNowPath ' >> myrun_UNN']);
    unix(['echo "srun vasp > log" >> myrun_UNN']);
    [a,b] = unix(['sbatch myrun_UNN '] );
    start_marker=findstr(b,'job ');
    jobNumber = b(start_marker(1)+4:end-1);
    disp([ 'Individual : ' num2str(Ind_No) ' -- JobID :', num2str(jobNumber) ]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

if isempty(jobNumber)
    jobNumber = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ORG_STRUC.platform > 0
    cd([ORG_STRUC.homePath,'/CalcFoldTemp']);
    ID=['.JOB-Gen',num2str(POP_STRUC.generation),'Ind',num2str(Ind_No),'Step',num2str(POP_STRUC.POPULATION(Ind_No).Step),'Fold',num2str(POP_STRUC.POPULATION(Ind_No).Folder)];
    [a,b]=unix(['touch ' num2str(jobNumber), ID]);
    
    ID=['Generation ' num2str(POP_STRUC.generation), ' Step ' num2str(num2str(POP_STRUC.POPULATION(Ind_No).Step)) ' of Structure ' num2str(Ind_No),' at Calcfold' num2str(POP_STRUC.POPULATION(Ind_No).Folder) ' : JOBID ' num2str(jobNumber) ];
    [a,b]=unix(['echo -n "' ID,'  Submitted " >> Jobs.history ' ]);
    [a,b]=unix(['echo -e `date +"%b%d-%T"` >> Jobs.history ' ]);
end
