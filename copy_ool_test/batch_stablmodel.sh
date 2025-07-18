#!/usr/bin/bash
#SBATCH --job-name=comp_7drugs_EGA2
#SBATCH --output=logs/ptb/comp_7drugs_EGA2%j.out
#SBATCH --error=logs/ptb/comp_7drugs_EGA2%j.err
#SBATCH --time=18:20:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB

module load python/3.9.0
source /home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/stabl_uni_clean/bin/activate

python run_regression_cv.py \
    --features_path /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/OSCC_Recurrence_features.csv \
    --results_dir /home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/jakob_cancer_logit_KO_GSS \
    --artificial_type knockoff \
    --model_chosen logit