#!/usr/bin/bash
#SBATCH --job-name=gridsearch_RMSE_XGB_OT
#SBATCH --output=logs/ptb/gridsearch_RMSE_XGB_OT%j.out
#SBATCH --error=logs/ptb/gridsearch_RMSE_XGB_OT%j.err
#SBATCH --time=2:20:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB

module load python/3.9.0
source ../../Stabl_venv/bin/activate

python ../grid_search_model.py \
    --features_path ../Data/merge_long_OOL_prediction.csv \
    --fold_feats_path ../Results/results_XGB_KO_merge_long_OOL_prediction \
    --results_dir ../Results/onset_test/gridsearch_RMSE_XGB_merged_EN \
    --artificial_type knockoff \
    --model_chosen xgboost