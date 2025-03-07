#!/bin/bash
#SBATCH -J QQP_SST2_MRPC_TESTING_50                                               
#SBATCH -A eecs   
#SBATCH -p dgx2                                                           
#SBATCH -o out/qqp_sst2_mrpc_50_out                           
#SBATCH -e out/qqp_sst2_mrpc_50_err                         

#SBATCH --time=7-00:00:00                 
#SBATCH --gres=gpu:1                         
#SBATCH --mem=64G                          

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=johnsga2@oregonstate.edu


module load python/3.10
module load cuda/12.1
module load openssl/3.1.5

bash sbatch_scripts/setup_qqp_sst2_mrpc_50.sh
