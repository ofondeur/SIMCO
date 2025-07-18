#!/usr/bin/bash
#SBATCH --job-name=gridsearch_RMSE_XGB_OT
#SBATCH --output=logs/ptb/gridsearch_RMSE_XGB_OT%j.out
#SBATCH --error=logs/ptb/gridsearch_RMSE_XGB_OT%j.err
#SBATCH --time=2:20:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB

module load python/3.9.0
source /home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/stabl_uni_clean/bin/activate

python grid_search_model.py \
    --features_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/copy_ool_test/ina_13OG_final_long_allstims_filtered.csv \
    --fold_feats_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/results_13OG_XGB_KO_168_filtered_allstim \
    --results_dir /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/gridsearch_RMSE_XGB_OT \
    --artificial_type knockoff \
    --model_chosen xgboost