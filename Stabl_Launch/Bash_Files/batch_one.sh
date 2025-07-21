#!/usr/bin/bash
#SBATCH --job-name=results_merge_OOL_prediction_KO_XGB
#SBATCH --output=logs/ptb/results_merge_OOL_prediction_KO_XGB%j.out
#SBATCH --error=logs/ptb/results_merge_OOL_prediction_KO_XGB%j.err
#SBATCH --time=2:20:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB

module load python/3.9.0
source Stabl_venv/bin/activate

python run_cv_existing_feats.py \
    --features_path ../Data/ina_13OG_final_long_allstims_filtered.csv \
    --fold_feats_path ../Results/results_merge_OOL_prediction_KO_XGB \
    --results_dir ../Results/merge_OOL_unstim_prediction_XGBdata \
    --artificial_type knockoff \
    --model_chosen xgboost