#!/usr/bin/bash
#SBATCH --job-name=logs_drugwise
#SBATCH --output=logs_unstim/train_cellot_drugwise_%A_%a.out
#SBATCH --error=logs_unstim/train_cellot_drugwise_%A_%a.err
#SBATCH --time=25:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=64GB
#SBATCH --array=0-251

module load python/3.9.0

# --- Activate Virtualenv ---
VENV_PATH="/home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/peter_ot/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# === CONFIGURATION ===
model='drug_OG13'
cv_condition='HVPV'
model_config='original'
stim='Unstim'

BASE_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype"
RESULTS_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_unstim/cross_validation_${model}"
JOB_LIST_FILE="${BASE_DIR}/job_list_all_drugs.txt"
PATIENT_FILE="${BASE_DIR}/patients.txt"
CONFIG_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/configs/tasks"
batch_corr_path="/home/groups/gbrice/ptb-drugscreen/ot/cellot/cross_validation/batchcorrection_2.csv"

mkdir -p "${RESULTS_DIR}"

# === Read job definition from file ===
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
JOB_DEF=$(sed -n "${LINE_NUM}p" "${JOB_LIST_FILE}")
IFS=',' read -r drug_used sanitized_celltype original_celltype <<< "${JOB_DEF}"

if [ -z "$drug_used" ] || [ -z "$sanitized_celltype" ]; then
    echo "ERROR: Missing job definition at line ${LINE_NUM}"
    exit 1
fi

echo "======================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Drug: ${drug_used}"
echo "Stim: ${stim} (fixed)"
echo "Sanitized Cell Type: ${sanitized_celltype}"
echo "Original Cell Type: ${original_celltype}"
echo "======================================================"

# === Read patients & split into folds ===
readarray -t all_patients < <(shuf "${PATIENT_FILE}")
n_patients=${#all_patients[@]}
n_folds=4
patients_per_fold=$(( (n_patients + n_folds - 1) / n_folds ))

declare -a folds
for (( i=0; i<n_folds; i++ )); do
    start_idx=$(( i * patients_per_fold ))
    folds[i]="${all_patients[@]:start_idx:patients_per_fold}"
done

for (( i=0; i<${n_folds}; i++ )); do
    folds[$i]=$(echo ${folds[$i]} | sed 's/ *$//g')
    echo "Fold ${i} patients: ${folds[$i]}"
done

# === Path definitions ===
JOB_NAME_FULL="${stim}_${sanitized_celltype}"
CONFIG_PATH_FULL="${CONFIG_DIR}/drug_model_${model_config}_${drug_used}/${cv_condition}/ptb_${JOB_NAME_FULL}_${cv_condition}_train.yaml"
OUTDIR_FULL="${RESULTS_DIR}/${drug_used}/${sanitized_celltype}/model-${JOB_NAME_FULL}"

mkdir -p "${OUTDIR_FULL}"
TEST_FILE="${OUTDIR_FULL}/cache/mmd_log.csv"

# === Run only if not already done ===
if [ ! -f "$TEST_FILE" ]; then
    echo "[INFO] Training for ${stim} / ${sanitized_celltype} / ${drug_used}"
    python /home/groups/gbrice/ptb-drugscreen/ot/cellot/scripts/train.py \
        --config /home/groups/gbrice/ptb-drugscreen/ot/cellot/configs/models/cellot_steroids.yaml \
        --config "${CONFIG_PATH_FULL}" \
        --outdir "${OUTDIR_FULL}" \
        --config.data.stim "${stim}" \
        --config.data.drug_used "${drug_used}" \
        --config.data.target "${drug_used}" \
        --config.data.batch_correction "${batch_corr_path}"
else
    echo "[INFO] Job already completed: ${TEST_FILE} exists. Skipping."
fi
