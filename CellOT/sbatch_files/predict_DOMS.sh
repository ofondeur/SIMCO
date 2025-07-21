#!/usr/bin/bash
#SBATCH --job-name=predict_surge_splcorr
#SBATCH --output=predict_surge_splcorr.%j.out
#SBATCH --error=predict_surge_splcorr.%j.err
#SBATCH --time=0:30:00
#SBATCH -p normal
#SBATCH -c 6
#SBATCH --mem=40GB

module load python/3.9.0

source cellot_sherlock_venv/bin/activate

python ./pred_CMC.py