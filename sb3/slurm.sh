#!/bin/bash
#SBATCH --job-name=mh
#SBATCH --nodes=1   
#SBATCH --partition=stampede         
#SBATCH --output=cluster_output.log      
#SBATCH --error=cluster_error.log      

# run the script
cd $SLURM_SUBMIT_DIR # where I ran sbatch from

source /home-mscluster/aboyley/anaconda3/etc/profile.d/conda.sh

# activate python virtual environment
conda activate minihack

python3 sb3_ppolstm_minihack.py

# deactivate python virtual environment
conda deactivate
