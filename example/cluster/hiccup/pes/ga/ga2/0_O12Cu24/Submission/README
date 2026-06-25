Running USPEX periodically


The real calculation starts with the command ‘matlab $<$ USPEX.m $>$ log’. Each time the MATLAB process will check the status of the running ab initio calculations. If the job is complete, MATLAB will go the the calculation folder to read the results, and then submit new calculations. After that, MATLAB will exit. Therefore, one needs to periodically call the command (for example, every 5 minutes). The periodic script can be executed by using either crontab or a shell script.

1) Crontab: This can be performed using a crontab daemon on your Linux machine. In your user home directory, there should now be the files:

  ~/call_job
  ~/CronTab

Here is an example of a 1-line CronTab file from one of our clusters:

  */5 * * * * sh call_job

It states that the interval between job submissions is 5 minutes and points to the file call_job, which should contain the address of the directory where USPEX will be executed, and the file call_job looks like this:


######################
#!/bin/tcsh

source /etc/csh.login
source ${HOME}/.cshrc

cd /ExecutionDirectory
date >> log
matlab < USPEX.m >> log
#######################

To activate crontab, type

  crontab ~/CronTab

If you want to terminate this run, either edit call_job or remove this crontab by typing

  crontab -r

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!To check if crontab works well, one should also keep tracking the updates of the log file at the beginning of the calculation.!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

2) Shell script: You can also prepare the script by using the sleep command in Linux shell. Below is a rather simple script run-uspex.sh:

############################
#!/bin/sh
while true
do
   date >> log 
   matlab < USPEX.m >> log
   sleep 300
done
###########################


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!Note: keep in mind that this calculation can only be terminated by killing the process ID of this script.!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

