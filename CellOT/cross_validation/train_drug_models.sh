#!/usr/bin/bash
#SBATCH --job-name=drug_OG13
#SBATCH --output=logs_drugs/drug_OG13/train_cellot_cv_%A_%a.out # Changed log name
#SBATCH --error=logs_drugs/drug_OG13/train_cellot_cv_%A_%a.err # Changed log name
#SBATCH --time=25:00:00 # Increased time slightly per job
#SBATCH -p normal
#SBATCH -c 1 # Maybe increase cores slightly if I/O or Python is heavy
#SBATCH --mem=64GB # Increased memory slightly
#SBATCH --array=0-167 #<--- UPDATE THIS based on output of generate_cellwise_joblist.py (e.g., 0-166 if 167 jobs)

module load python/3.9.0

VENV_PATH="/home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/peter_ot/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

model='drug_OG13'
cv_condition='HVPV'
model_config='original'
drug_used='PRA'

# --- Directory Definitions ---
BASE_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype"
RESULTS_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_drug/cross_validation_${model}/${drug_used}"
JOB_LIST_FILE="${BASE_DIR}/valid_jobs.txt"
PATIENT_FILE="${BASE_DIR}/patients.txt"
CONFIG_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/configs/tasks"
mkdir -p "${RESULTS_DIR}"

# --- Read Job Assignment ---
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
JOB_DEF=$(sed -n "${LINE_NUM}p" "${JOB_LIST_FILE}")

if [ -z "$JOB_DEF" ]; then
    echo "ERROR: Could not read job definition for task ID ${SLURM_ARRAY_TASK_ID} (line ${LINE_NUM}) from ${JOB_LIST_FILE}"
    exit 1
fi

IFS=',' read -r stim sanitized_celltype original_celltype <<< "${JOB_DEF}"

echo "======================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Assigned Stim: ${stim}"
echo "Assigned Cell Type (Sanitized): ${sanitized_celltype}"
echo "Assigned Cell Type (Original): ${original_celltype}"
echo "======================================================"

if [ ! -f "${PATIENT_FILE}" ]; then
  echo "ERROR: Patient file ${PATIENT_FILE} not found!" >&2
  exit 1
fi
readarray -t all_patients < <(shuf "${PATIENT_FILE}")

n_patients=${#all_patients[@]}
n_folds=4
patients_per_fold=$(( (n_patients + n_folds - 1) / n_folds ))

echo "Total patients: ${n_patients}"
if [ $((n_patients % n_folds)) -ne 0 ]; then
    echo "WARNING: Number of patients (${n_patients}) is not perfectly divisible by n_folds (${n_folds}). Folds might be slightly uneven."
fi

declare -a folds


for (( i=0; i<n_folds; i++ )); do
    start_idx=$(( i * patients_per_fold ))
    folds[i]="${all_patients[@]:start_idx:patients_per_fold}"
done


for (( i=0; i<${n_folds}; i++ )); do
    folds[$i]=$(echo ${folds[$i]} | sed 's/ *$//g')
    echo "Fold ${i} patients: ${folds[$i]}"
done

batch_corr_path="/home/groups/gbrice/ptb-drugscreen/ot/cellot/cross_validation/batchcorrection_2.csv"
JOB_NAME_FULL="${stim}_${sanitized_celltype}"
CONFIG_PATH_FULL="${CONFIG_DIR}/drug_model_${model_config}_${drug_used}/${cv_condition}/ptb_${JOB_NAME_FULL}_${cv_condition}_train.yaml"
OUTDIR_FULL="${RESULTS_DIR}/${stim}/${sanitized_celltype}/model-${JOB_NAME_FULL}"

mkdir -p "${OUTDIR_FULL}"
TEST_FILE="${OUTDIR_FULL}/cache/model.pt"

if [ ! -f "$TEST_FILE" ]; then

    echo "[INFO]  Full Model for ${stim} / ${sanitized_celltype} "
    python /home/groups/gbrice/ptb-drugscreen/ot/cellot/scripts/train.py \
          --config /home/groups/gbrice/ptb-drugscreen/ot/cellot/configs/models/cellot_steroids.yaml \
          --config "${CONFIG_PATH_FULL}" \
          --outdir "${OUTDIR_FULL}" \
          --config.data.stim "${stim}" \
          --config.data.drug_used "${drug_used}" \
          --config.data.target "${drug_used}" \
          --config.data.batch_correction "${batch_corr_path}"
fi

echo "Job for Stim=${stim}, CellType=${celltype} completed."