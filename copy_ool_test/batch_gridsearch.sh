#!/usr/bin/bash
#SBATCH --job-name=gridsearch_RMSE_XGB_OT_v2
#SBATCH --output=logs/ptb/gridsearch_RMSE_XGB_OT_v2%j.out
#SBATCH --error=logs/ptb/gridsearch_RMSE_XGB_OT_v2%j.err
#SBATCH --time=2:20:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB

module load python/3.9.0
source /home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/stabl_uni_clean/bin/activate

python grid_search_model.py \
    --features_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/ina_13OG_df_168_filtered_allstim_new.csv \
    --fold_feats_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/results_13OG_XGB_KO_168_filtered_allstim_new \
    --results_dir /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/gridsearch_RMSE_XGB_OT_v2 \
    --artificial_type knockoff \
    --model_chosen xgboost