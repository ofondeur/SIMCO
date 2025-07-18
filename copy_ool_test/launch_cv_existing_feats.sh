#!/usr/bin/bash
#SBATCH --job-name=ina_13OG_df_168_filtered_no_unstim_new
#SBATCH --output=logs/ptb/ina_13OG_df_168_filtered_no_unstim_new%j.out
#SBATCH --error=logs/ptb/ina_13OG_df_168_filtered_no_unstim_new%j.err
#SBATCH --time=13:50:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=32GB

module load python/3.9.0
source /home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/stabl_uni_clean/bin/activate

python run_regression_cv.py \
    --features_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/ina_13OG_df_168_filtered_allstim_new.csv \
    --results_dir /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/ina_13OG_df_168_filtered_no_unstim_new \
    --artificial_type knockoff \
    --model_chosen xgboost