#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 12
#SBATCH --constraint="intel"
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=100MB
#SBATCH --partition=plgrid
#SBATCH --account=plgmpatyk2020a

module load plgrid/tools/python-intel/3.6.5

for n in 1000 10000 100000; do
  echo "$n"
    for ((i=1; i<=$1; i++)); do
    echo "$i"
      for ((j=1; j<=$2; j++)); do
        mpirun -np "$i" ./pcam_parallel_stripes.py "$3" "$n" "$i"
      done
    done
done

# $1 - max tasks
# $2 - computation repetitions
# $3 - iterations
# sbatch batch.sh 12 3 100