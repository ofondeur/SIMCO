#!/usr/bin/bash
#SBATCH --job-name=predict_unstim_drug_ina
#SBATCH --output=logs/predict_unstim_drug_ina.%j.out
#SBATCH --error=logs/predict_unstim_drug_ina.%j.err
#SBATCH --time=08:45:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=32GB

module load python/3.9.0

source /home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/peter_ot/bin/activate
python /home/groups/gbrice/ptb-drugscreen/ot/cellot/make_prediction/predict_unstim_drug_ina.py