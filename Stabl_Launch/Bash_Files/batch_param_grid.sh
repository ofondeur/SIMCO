#!/usr/bin/bash
#SBATCH --job-name=GS_XGB
#SBATCH --output=logs/ptb_grid/gridsearch_xgb_%A_%a.out
#SBATCH --error=logs/ptb_grid/gridsearch_xgb_%A_%a.err
#SBATCH --time=0:20:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --array=0-95

module load python/3.9.0
source ../../Stabl_venv/bin/activate

PARAM_FILE="param_grid_files/params_${SLURM_ARRAY_TASK_ID}.json"

python ../run_cv_existing_feats.py \
    --notreat_features_path ../Data/ina_13OG_final_long_allstims_filtered.csv \
    --fold_feats_path ../Results/results_ina_13OG_final_long_allstims_filtered_xgboost_knockoff_GroupShuffleSplit \
    --results_dir ../Results/comp_9drugs_long/run_${SLURM_ARRAY_TASK_ID} \
    --artificial_type knockoff \
    --model_chosen xgboost \
    --xgb_config_path $PARAM_FILE
