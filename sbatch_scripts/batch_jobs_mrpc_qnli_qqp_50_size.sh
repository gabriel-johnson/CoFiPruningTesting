#!/bin/bash
#SBATCH -J SIZE                                               
#SBATCH -A eecs   
#SBATCH -p dgx2                                                           
#SBATCH -e out/mrpc_qnli_qqp_50_size_err                            
#SBATCH -o out/mrpc_qnli_qqp_50_size_out                            

#SBATCH --time=3-00:00:00                 
#SBATCH --gres=gpu:1                         
#SBATCH --mem=128G                          

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=johnsga2@oregonstate.edu


module load python/3.10
module load cuda/12.1
module load openssl/3.1.5

bash sbatch_scripts/setup_mrpc_qnli_qqp_50_size.sh
