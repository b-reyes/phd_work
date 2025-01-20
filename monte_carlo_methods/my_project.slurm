#!/bin/bash -x
#SBATCH --job-name="BR_thesis_code"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks=16
#SBATCH --exclusive                 
#SBATCH --export=ALL
#SBATCH --account=150315205011 
#SBATCH --time=1:00:00
#SBATCH -o con.out
#SBATCH -e con.err

var=$1
var2=$2

if (($var == 0)) && (($var2 == 0)); then
    srun ./onedmc_exe 
elif (($var == 1)) && (($var2 == 0)); then
    srun ./twodmc_exe
elif (($var == 2)) && (($var2 == 0)); then
    srun ./threedmc_exe
elif (($var == 0)) && (($var2 == 1)); then
    srun ./onedqmc_exe
elif (($var == 1)) && (($var2 == 1)); then
    srun ./twodqmc_exe
elif (($var == 2)) && (($var2 == 1)); then
    srun ./threedqmc_exe
elif (($var == 0)) && (($var2 == 2)); then
    srun ./onedmlmc_exe
elif (($var == 1)) && (($var2 == 2)); then
    srun ./twodmlmc_exe
elif (($var == 2)) && (($var2 == 2)); then
    srun ./threedmlmc_exe
elif (($var == 0)) && (($var2 == 3)); then
    srun ./onedmlqmc_exe
elif (($var == 1)) && (($var2 == 3)); then
    srun ./twodmlqmc_exe
elif (($var == 2)) && (($var2 == 3)); then
    srun ./threedmlqmc_exe
elif (($var == 0)) && (($var2 == 4)); then
    srun ./onedmlmctst_exe
elif (($var == 1)) && (($var2 == 4)); then
    srun ./twodmlmctst_exe
elif (($var == 2)) && (($var2 == 4)); then
    srun ./threedmlmctst_exe
fi

cd ..

cd run_dir 

cp con.out con.err ../results

cd ..

rm -r run_dir

mkdir run_dir



