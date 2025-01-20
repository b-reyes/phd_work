#!/bin/bash -x
#SBATCH --job-name="AC3dmc"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --exclusive                                         
#SBATCH --export=ALL
#SBATCH --account=150315205011
#SBATCH --time=1:00:00
#SBATCH -o con.out
#SBATCH -e con.err

srun ./my_exe
