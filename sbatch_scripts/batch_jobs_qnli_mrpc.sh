#!/bin/bash
#SBATCH -J QNLI_MRPC_50                                               
#SBATCH -A eecs   
#SBATCH -p dgx2                                                           
#SBATCH -o out/qnli_mrpc_req_grad_out                            
#SBATCH -e out/qnli_mrpc_req_grad_out                            

#SBATCH --time=7-00:00:00                 
#SBATCH --gres=gpu:1                         
#SBATCH --mem=128G                          

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=johnsga2@oregonstate.edu


module load python/3.10
module load cuda/12.1
module load openssl/3.1.5

bash sbatch_scripts/setup_qnli_mrpc_50.sh
