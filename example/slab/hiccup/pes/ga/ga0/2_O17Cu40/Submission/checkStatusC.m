function doneOr = checkStatusC(Ind_No)

% USPEX Version 6.2
% Change: Added remote functionality
global POP_STRUC
global ORG_STRUC

jobID=POP_STRUC.POPULATION(Ind_No).JobID;
Step = POP_STRUC.POPULATION(Ind_No).Step;

% doneOr should equal to 1 if the job is finished
doneOr = 0;

cd ([ORG_STRUC.homePath '/CalcFold' num2str(POP_STRUC.POPULATION(Ind_No).Folder)])

if jobID == 0
    not = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif jobID == 0.01    % used for reoptOld = 0 
    doneOr = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif jobID == 0.02    % absence of optimization
    doneOr = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform==0 %nonP
    if ORG_STRUC.numParallelCalcs > 1
       [a,b] = unix('grep "JOB_IS_FINISHED" JOB_LOG');
       if ~isempty(findstr(b,'JOB_IS_FINISHED'))
          disp(['JOB is DONE']);
          [nothing, nothing] = unix('rm JOB_LOG');
          doneOr = 1;
       else
          doneOr = 0;
       end
    else
       doneOr = 1;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform == 1 %from USER localsubmission
    doneOr = checkStatus_local(jobID);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform == 2 %from USER remote submission
    doneOr = checkStatus_remote(jobID, ORG_STRUC.remoteFolder, POP_STRUC.POPULATION(Ind_No).Folder);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform==6 %SIESTAlocal
    [a,b] = unix('grep "End of run" output');
    c = findstr(b,'End');
    if ~isempty(c)
     doneOr = 1;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform==3 | ORG_STRUC.platform==9 %CFN
    [a,b] = unix(['qstat ' jobID '']);
    if isempty(findstr(b,'R cfn_')) & isempty(findstr(b,'Q cfn_'))
       doneOr = 1;
       [nothing, nothing] = unix('rm USPEX*');    % to remove the log file
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform==4 %QSH 
    [nothing, statusStr] = unix(['bjobs ' num2str(jobID) ]);
    doneOr = strfind(statusStr, 'found');

    if ~isempty(doneOr)
        doneOr = 1;
    else
        doneOr  = strfind(statusStr,'DONE');   % done jobs
        doneOr1 = strfind(statusStr,'EXIT');   % deleted jobs
        doneOr2 = strfind(statusStr,'UNKWN');  % dead calculations
        doneOr3 = strfind(statusStr,'ZOMBI');  % dead calculations
        if ~isempty(doneOr) | ~isempty(doneOr1) | ~isempty(doneOr2) | ~isempty(doneOr3)
            doneOr = 1;
            %[nothing, nothing] = unix('rm output_*');    % to remove the log file
        else
            doneOr = 0;
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform==5 %QSH2 
    [nothing, statusStr] = unix(['bjobs ' num2str(jobID) ]);
    doneOr = strfind(statusStr, 'found');

    if ~isempty(doneOr)
        doneOr = 1;
    else
        doneOr  = strfind(statusStr,'DONE');
        doneOr1 = strfind(statusStr,'EXIT');
        doneOr2 = strfind(statusStr,'UNKWN');  % dead calculations
        doneOr3 = strfind(statusStr,'ZOMBI');  % dead calculations
        if ~isempty(doneOr) | ~isempty(doneOr1) | ~isempty(doneOr2) | ~isempty(doneOr3)
            doneOr = 1;
%            [nothing, nothing] = unix('rm output_*');    % to remove the log file
        else
            doneOr = 0;
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif ORG_STRUC.platform==6  %xservDE
    [nothing, statusStr] = unix(['qstat -j ' num2str(jobID) ]);
    d = strfind(statusStr, 'exist');
    if ~isempty(d)
        doneOr = 1;
    end
elseif ORG_STRUC.platform==7  %MIPT
    [nothing, nothing] = unix(['squeue > '  ORG_STRUC.homePath '/jobinfo.dat']);
    [nothing, statusStr] = unix(['cat ' ORG_STRUC.homePath '/jobinfo.dat | grep -v CG | grep -v JOBID | grep ' num2str(jobID) ' | wc -l '] );
    [nothing, nothing] = unix(['cat ' ORG_STRUC.homePath '/jobinfo.dat | grep -v CG | grep -v JOBID | grep ' num2str(jobID) ' '] )
    disp(statusStr)
    isOK=str2num( statusStr );

    if  isOK==0
        doneOr = 1;
    else
        doneOr = 0;
    end
    [a,b] = unix(['qstat ' jobID ''])


elseif ORG_STRUC.platform==8  %NWPU
    [a,b] = unix(['qstat ' jobID ''])
    if isempty(findstr(b,'R batch')) & isempty(findstr(b,'Q batch'))
       doneOr = 1
       [nothing, nothing] = unix(['rm *' jobID ]);    % to remove the log file
    else
       doneOr = 0;
    end
elseif ORG_STRUC.platform == 10  % UNN
    unix(['squeue > '  ORG_STRUC.homePath '/jobinfo.dat']);
    [nothing, statusStr] = unix(['cat ' ORG_STRUC.homePath '/jobinfo.dat | grep -v CG | grep -v JOBID | grep ' num2str(jobID) ' | wc -l '] );
    unix(['cat ' ORG_STRUC.homePath '/jobinfo.dat | grep -v CG | grep -v JOBID | grep ' num2str(jobID) ' '] );
    %disp(statusStr);
    isOK=str2num( statusStr );
    if  isOK==0
        doneOr = 1;
    else
        doneOr = 0;
    end
    [a,b] = unix(['qstat ' jobID ''])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ORG_STRUC.platform>0  & doneOr==1
     [nothing, nothing] = unix(['rm ' ORG_STRUC.homePath '/CalcFoldTemp/' num2str(jobID),'.*']);
end

cd (ORG_STRUC.homePath)
