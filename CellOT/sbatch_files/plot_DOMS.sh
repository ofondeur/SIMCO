#!/usr/bin/bash
#SBATCH --job-name=plot_surgeunscorr
#SBATCH --output=plot_surgeunscorr.%j.out
#SBATCH --error=plot_surgeunscorr.%j.err
#SBATCH --time=1:00:00
#SBATCH -p normal
#SBATCH -c 6
#SBATCH --mem=40GB

module load python/3.9.0

source cellot_sherlock_venv/bin/activate

python ./plot_pred_vs_true_uns_corr.py