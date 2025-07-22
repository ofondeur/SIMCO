#!/usr/bin/bash
#SBATCH --job-name=cross_validation_KO_XGB
#SBATCH --output=logs/ptb/cross_validation_KO_XGB%j.out
#SBATCH --error=logs/ptb/cross_validation_KO_XGB%j.err
#SBATCH --time=13:50:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=32GB

module load python/3.9.0
source ../../Stabl_venv/bin/activate

python ../run_regression_cv.py \
    --features_path ../Data/ina_13OG_final_long_allstims_filtered.csv \
    --results_dir ../Results/cross_validation_KO_XGB \
    --artificial_type knockoff \
    --model_chosen xgboost