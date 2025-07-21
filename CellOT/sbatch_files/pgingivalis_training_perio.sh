#!/usr/bin/bash
#SBATCH --job-name=train_cellot_dual
#SBATCH --output=logs/train_cellot_dual_%A_%a.out
#SBATCH --error=logs/train_cellot_dual_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH --array=1-15

module load python/3.9.0
source cellot_sherlock_venv/bin/activate

# Lire le couple Cellule-Stimulation depuis le fichier
PAIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pgingivalis_percelltype_perio.txt)
STIM=$(echo "$PAIR" | awk '{print $1}')
CELL=$(echo "$PAIR" | awk '{$1=""; print $0}' | sed 's/^ //')

SAFE_CELL=$(echo "$CELL" | tr ' ' '_')
STIM_Esp=$(echo "$STIM" | tr '_' ' ')

# Boucle sur les deux types : doms et surge
for TYPE in doms surge; do
  CONFIG_PATH="./configs/tasks/sherlock_perio_${TYPE}_corrected/perio_data_sherlock${STIM}_${SAFE_CELL}_train.yaml"

  if [ ! -f "$CONFIG_PATH" ]; then
      echo "Config file not found: $CONFIG_PATH"
      continue
  fi

  OUTDIR="./results/perio_${TYPE}_corrected_training/${STIM}_${CELL}/model-cellot"

  echo "Launching training for $TYPE"
  echo "STIM: $STIM"
  echo "CELL: $CELL"
  echo "CONFIG_PATH: $CONFIG_PATH"
  echo "OUTDIR: $OUTDIR"

  python ./scripts/train.py \
    --outdir "$OUTDIR" \
    --config "$CONFIG_PATH" \
    --config ./configs/models/cellot.yaml \
    --config.data.target "$STIM_Esp"
done
