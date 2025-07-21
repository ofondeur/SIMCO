#!/usr/bin/bash
#SBATCH --job-name=original_diffIO_med_HVPV
#SBATCH --output=logs/original_diffIO_med_HVPV/train_cellot_cv_%A_%a.out # Changed log name
#SBATCH --error=logs/original_diffIO_med_HVPV/train_cellot_cv_%A_%a.err # Changed log name
#SBATCH --time=9:00:00 # Increased time slightly per job
#SBATCH -p normal
#SBATCH -c 4 # Maybe increase cores slightly if I/O or Python is heavy
#SBATCH --mem=128GB # Increased memory slightly
#SBATCH --array=0-167 #<--- UPDATE THIS based on output of generate_cellwise_joblist.py (e.g., 0-166 if 167 jobs)

module load python/3.9.0

# --- IMPORTANT: Verify your virtual environment path ---
VENV_PATH="/home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/peter_ot/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi
# Check if python is working
python -c "import sys; print(f'Python executable: {sys.executable}')"

model='original_diffIO_med_PV' # choose among 'olivier', 'peter', 'original','all_markers' 

# --- Directory Definitions ---
BASE_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype"
RESULTS_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_${model}"
JOB_LIST_FILE="${BASE_DIR}/valid_jobs.txt" #<-- Path to job list created by pre-script

CONFIG_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/configs/tasks"
mkdir -p "${RESULTS_DIR}"

# --- Read Job Assignment ---
# Read the specific line corresponding to the SLURM task ID (array index + 1)
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1)) # Calculate line number first
JOB_DEF=$(sed -n "${LINE_NUM}p" "${JOB_LIST_FILE}") # Use the calculated line number
# --- CHANGE END ---
if [ -z "$JOB_DEF" ]; then
    echo "ERROR: Could not read job definition for task ID ${SLURM_ARRAY_TASK_ID} (line ${LINE_NUM}) from ${JOB_LIST_FILE}"
    exit 1
fi
# --- Read three fields ---
IFS=',' read -r stim sanitized_celltype original_celltype <<< "${JOB_DEF}"
# Note: celltype read from the file should already be sanitized

echo "======================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}, Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Assigned Stim: ${stim}"
echo "Assigned Cell Type (Sanitized): ${sanitized_celltype}"
echo "Assigned Cell Type (Original): ${original_celltype}" # Print original too
echo "======================================================"


cv_condition='HVPV'
model_config='different_IO'
drug_used='DMSO'
batch_corr_path="/home/groups/gbrice/ptb-drugscreen/ot/cellot/cross_validation/batchcorrection.csv"
JOB_NAME_FULL="${stim}_${sanitized_celltype}"
CONFIG_PATH_FULL="${CONFIG_DIR}/ptb_final_cv_${model_config}/${cv_condition}/ptb_${JOB_NAME_FULL}_${cv_condition}_train.yaml"
OUTDIR_FULL="${RESULTS_DIR}/${stim}/${sanitized_celltype}/model-${JOB_NAME_FULL}"
mkdir -p "${OUTDIR_FULL}"
TEST_FILE="${OUTDIR_FULL}/cache/model.pt"

#check if the model was trained before by checking if a .pt was saved

if [ ! -f "$TEST_FILE" ]; then

    echo "[INFO] Relancing Full Model for ${stim} / ${sanitized_celltype} (missing ${TEST_FILE})"
    python /home/groups/gbrice/ptb-drugscreen/ot/cellot/scripts/train.py \
          --config /home/groups/gbrice/ptb-drugscreen/ot/cellot/configs/models/cellot.yaml \
          --config "${CONFIG_PATH_FULL}" \
          --outdir "${OUTDIR_FULL}" \
          --config.data.target "${stim}" \
          --config.data.drug_used "${drug_used}"
fi

