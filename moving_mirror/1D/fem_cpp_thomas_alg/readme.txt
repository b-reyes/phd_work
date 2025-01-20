to run: 
1. type make 
2. type ./my_exe to run 

MODULES TO LOAD ON HPC SYSTEM MIO:
module purge 
module load PrgEnv/devtoolset-9
module load PrgEnv/intel/latest

srun -N 1 --ntasks-per-node=28 --time=30:05:00 --pty bash
