#!/usr/bin/bash
#SBATCH --job-name=ptb_different_IO_batchcorr
#SBATCH --output=logs/ptb_different_IO_batchcorr/train_cellot_cv_%A_%a.out
#SBATCH --error=logs/ptb_different_IO_batchcorr/train_cellot_cv_%A_%a.err
#SBATCH --time=40:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=32GB
#SBATCH --array=0-167
module load python/3.9.0

VENV_PATH="../../test_venv-cellot2/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

model='different_IO' 

BASE_DIR="../datasets/ptb_concatenated_per_condition_celltype"
RESULTS_DIR="../results/cross_validation_${model}_batchcorr"
JOB_LIST_FILE="${BASE_DIR}/valid_jobs.txt"
PATIENT_FILE="${BASE_DIR}/patients_HV.txt"
CONFIG_DIR="../configs/tasks"
mkdir -p "${RESULTS_DIR}"

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
JOB_DEF=$(sed -n "${LINE_NUM}p" "${JOB_LIST_FILE}")
# --- CHANGE END ---
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

# --- Train FULL Model ---
echo "--- Training FULL Model ---"

cv_condition='HVPV'
drug_used='DMSO'
batch_corr_path="../cross_validation/batchcorrection.csv"
JOB_NAME_FULL="${stim}_${sanitized_celltype}"
CONFIG_PATH_FULL="${CONFIG_DIR}/ptb_final_cv_${model}/${cv_condition}/ptb_${JOB_NAME_FULL}_${cv_condition}_train.yaml"
OUTDIR_FULL="${RESULTS_DIR}/${stim}/${sanitized_celltype}/model-${JOB_NAME_FULL}"
mkdir -p "${OUTDIR_FULL}"
python ../scripts/train.py \
      --config ../configs/models/cellot_steroids.yaml \
      --config "${CONFIG_PATH_FULL}" \
      --outdir "${OUTDIR_FULL}" \
      --config.data.target "${stim}" \
      --config.data.drug_used "${drug_used}" \
      --config.data.batch_correction "${batch_corr_path}"


# --- Train 4-Fold CV Models ---
echo "--- Training 4-Fold CV Models ---"
for k in $(seq 0 $((n_folds - 1))); do
    echo "--- Processing Fold ${k} ---"
    # Identify patients *excluded* in this fold (test set)
    excluded_patients_str="${folds[$k]}"

    JOB_NAME_FOLD="${stim}_${sanitized_celltype}_fold${k}"
    OUTDIR_FOLD="${RESULTS_DIR}/${stim}/${sanitized_celltype}/model-${JOB_NAME_FOLD}"
    mkdir -p "${OUTDIR_FOLD}"
    echo "Saving excluded patients for Fold ${k} to ${OUTDIR_FOLD}/test_patients.txt"
    echo "${excluded_patients_str}" > "${OUTDIR_FOLD}/test_patients.txt"
    excluded_list_formatted="['$(echo ${excluded_patients_str} | sed "s/ /','/g")']"
    echo "Excluding patients: ${excluded_list_formatted}"

    # Launch training for this fold
    echo "Launching training for Fold ${k} (${JOB_NAME_FOLD})..."
    echo "Outdir for the fold is (${OUTDIR_FOLD})"
    python ../scripts/train.py \
      --config ../configs/models/cellot_steroids.yaml \
      --config "${CONFIG_PATH_FULL}" \
      --outdir "${OUTDIR_FOLD}" \
      --config.data.target "${stim}" \
      --config.data.patients_excluded "${excluded_list_formatted}" \
      --config.data.drug_used "${drug_used}" \
      --config.data.batch_correction "${batch_corr_path}"

    if [ $? -eq 0 ]; then
        echo "Fold ${k} training completed. Model saved to ${OUTDIR_FOLD}."
    else
        echo "ERROR: Fold ${k} training failed."
    fi

done

echo "Job for Stim=${stim}, CellType=${celltype} completed."