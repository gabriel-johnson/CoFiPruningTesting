#!/bin/bash
#SBATCH -J MRPC_QNLI_QQP_TESTING_50                                               
#SBATCH -A eecs   
#SBATCH -p dgx2                                                           
#SBATCH -e out/mrpc_qnli_qqp_50_err                            
#SBATCH -o out/mrpc_qnli_qqp_50_out                            

#SBATCH --time=7-00:00:00                 
#SBATCH --gres=gpu:1                         
#SBATCH --mem=128G                          

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=johnsga2@oregonstate.edu


module load python/3.10
module load cuda/12.1
module load openssl/3.1.5

bash sbatch_scripts/setup_mrpc_qnli_qqp_50.sh
