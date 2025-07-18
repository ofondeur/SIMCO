#!/usr/bin/bash
#SBATCH --job-name=comp_9drugs_long
#SBATCH --output=logs/ptb/comp_9drugs_long%j.out
#SBATCH --error=logs/ptb/comp_9drugs_long%j.err
#SBATCH --time=0:20:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB

module load python/3.9.0
source /home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/stabl_uni_clean/bin/activate

python run_cv_existing_feats.py \
    --notreat_features_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/ina_13OG_final_long_allstims_filtered.csv \
    --fold_feats_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/results_ina_13OG_final_long_allstims_filtered_xgboost_knockoff_GroupShuffleSplit \
    --results_dir /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/comp_9drugs_long \
    --artificial_type knockoff \
    --model_chosen xgboost