#!/bin/bash
#SBATCH -J BOTH_95                                              
#SBATCH -A eecs   
#SBATCH -p dgx2                                                           
#SBATCH -o out/qqp_qnli_mrpc_50_pre_out                           
#SBATCH -e out/qqp_qnli_mrpc_50_pre_err                         

#SBATCH --time=1-00:00:00                 
#SBATCH --gres=gpu:1                         
#SBATCH --mem=64G                          

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=johnsga2@oregonstate.edu


module load python/3.10
module load cuda/12.1
module load openssl/3.1.5

bash sbatch_scripts/setup_qqp_qnli_mrpc_95_both.sh
