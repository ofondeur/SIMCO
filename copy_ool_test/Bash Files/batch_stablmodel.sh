#!/usr/bin/bash
#SBATCH --job-name=run_stabl_XGB_KO_ptb_data
#SBATCH --output=logs/ptb/run_stabl_XGB_KO_ptb_data%j.out
#SBATCH --error=logs/ptb/run_stabl_XGB_KO_ptb_data%j.err
#SBATCH --time=13:20:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB

module load python/3.9.0
source /home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/stabl_uni_clean/bin/activate

python run_regression_cv.py \
    --features_path ../Data/ina_13OG_final_long_allstims_filtered.csv \
    --results_dir ../Results/run_stabl_XGB_KO_ptb_data \
    --artificial_type knockoff \
    --model_chosen xgboost