#!/usr/bin/bash
#SBATCH --job-name=cmcs_LPS_perio_dblcorr_outputmodified_noeval
#SBATCH --output=cmcs_LPS_perio_dblcorr_outputmodified_noeval.%j.out
#SBATCH --error=cmcs_LPS_perio_dblcorr_outputmodified_noeval.%j.err
#SBATCH --time=3:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB

module load python/3.9.0

source cellot_sherlock_venv/bin/activate

python ./scripts/train.py \
  --outdir ./results/cmcs_LPS_dblcorr_inputmodif_noeval/model-cellot \
  --config "./configs/tasks/perio_surge_train_dbl_corr/perio_data_sherlock_P._gingivalis_Classical_Monocytes_(CD14+CD16-)_train2.yaml" \
  --config ./configs/models/cellot.yaml \
  --config.data.target 'P. gingivalis'
