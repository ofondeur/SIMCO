#!/usr/bin/bash
#SBATCH --job-name=ot_features_13OG_cellwise_medians
#SBATCH --output=logs/ptb/ot_features_13OG_cellwise_medians%j.out
#SBATCH --error=logs/ptb/ot_features_13OG_cellwise_medians%j.err
#SBATCH --time=24:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=96GB

module load python/3.9.0
source /home/groups/gbrice/ptb-drugscreen/ot/cellot_pheno/cells_combined/stabl_uni/bin/activate

python run_regression_cv.py \
    --features_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/ot_features_13OG_cellwise_medians.csv \
    --results_dir /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/results_grouped_cv_13OG_cellwise_masked \
    --artificial_type knockoff