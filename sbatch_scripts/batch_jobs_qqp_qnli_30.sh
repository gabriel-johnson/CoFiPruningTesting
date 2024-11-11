#!/bin/bash
#SBATCH -J QQP_QNLI_TESTING                                               
#SBATCH -A eecs   
#SBATCH -p dgxh                                                           
#SBATCH -o out/qqp_qnli_30_out                            
#SBATCH -e out/qqp_qnli_30_err                            

#SBATCH --time=2-00:00:00                 
#SBATCH --gres=gpu:1                         
#SBATCH --mem=128G                          

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=johnsga2@oregonstate.edu


module load python/3.10
module load cuda/12.1
module load openssl/3.1.5

bash setup_qqp_qnli_30.sh
