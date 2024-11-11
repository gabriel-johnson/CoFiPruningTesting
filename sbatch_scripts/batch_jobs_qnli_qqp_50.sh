#!/bin/bash
#SBATCH -J QNLI_QQP_TESTING_50                                               
#SBATCH -A eecs   
#SBATCH -p dgxh                                                           
#SBATCH -o out/qnli_qqp_50_out                            
#SBATCH -e out/qnli_qqp_50_err                            

#SBATCH --time=2-00:00:00                 
#SBATCH --gres=gpu:1                         
#SBATCH --mem=128G                          

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=johnsga2@oregonstate.edu


module load python/3.10
module load cuda/12.1
module load openssl/3.1.5

bash sbatch_scripts/setup_qnli_qqp_50.sh
