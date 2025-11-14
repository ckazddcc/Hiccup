# Author: Dingming Chen
# Date: 2021.3.8


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=gpu_id
# running LAMMPS
mpirun -np 1 lmp -in lammps.in 1>run.log 2>err &

#PID=$!
#echo $PID > pid.txt


