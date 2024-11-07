#!/bin/bash
#SBATCH -J QQP_QNLI_TESTING						  # name of job
#SBATCH -A STARLAB	  # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p dgxh,share								  # name of partition or queue
#SBATCH -o out				  # name of output file for this submission script
#SBATCH -e err				  # name of error file for this submission script

#SBATCH --time=2-12:00:00                 # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
#SBATCH --gres=gpu:1                         # number of GPUs to request (default 0)
#SBATCH --mem=64G                          # request 10 gigabytes memory (per node, default depends on node)


# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load python/3.10
module load cuda/12.1
module load openssl/3.1.5

# run my job (e.g. matlab, python)
bash run.sh