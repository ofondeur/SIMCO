#!/usr/bin/bash
#SBATCH --job-name=create_data
#SBATCH --output=create_dataperio.%j.out
#SBATCH --error=create_dataperio.%j.err
#SBATCH --time=1:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB

module load python/3.9.0

source ~/flowcyto/bin/activate

python ./from_fcs_to_data.py