#!/usr/bin/bash
#SBATCH --job-name=train_cellot_param_CK
#SBATCH --output=logs/train_cellot_param_CK%A_%a.out
#SBATCH --error=logs/train_cellot_param_CK_%A_%a.err
#SBATCH --time=4:00:00
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
CONFIG_PATH="./configs/tasks/sherlock_perio/perio_data_sherlock${STIM}_${SAFE_CELL}_train.yaml"

# Vérifier que le fichier existe
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config file not found: $CONFIG_PATH"
    exit 1
fi

# Définir l'outdir en fonction de Cellule-Stimulation
OUTDIR="./results/perio_training_LR2/${STIM}_${CELL}/model-cellot"

echo "STIM: $STIM"
echo "CELL: $CELL"
echo "SAFE_CELL: $SAFE_CELL"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "OUTDIR: $OUTDIR"

# Lancer la commande avec les bons paramètres
python ./scripts/train.py \
  --outdir "$OUTDIR" \
  --config "$CONFIG_PATH" \
  --config ./configs/models/cellot_LR.yaml \
  --config.data.target "$STIM_Esp"