#!/usr/bin/bash

# Parameters
MODELS=("xgboost")
ARTIFICIALS=("knockoff")
FEATURES=("ina_13OG_final_long_allstims_filtered.csv")

# Paths
BASE_DIR="../"
SCRIPT_DIR="${BASE_DIR}/jobs"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$SCRIPT_DIR" "$LOG_DIR"

# Loop through all combinations
for model in "${MODELS[@]}"; do
  for artificial in "${ARTIFICIALS[@]}"; do
    for feature in "${FEATURES[@]}"; do

      feature_base=$(basename "$feature" .csv)
      job_name="${feature_base}_${model}_${artificial}_GSS"
      result_dir="${BASE_DIR}/Results/results_${job_name}"
      features_path="${BASE_DIR}/Data/${feature}"
      script_path="${SCRIPT_DIR}/job_${job_name}.sh"
      log_out="${LOG_DIR}/${job_name}_%j.out"
      log_err="${LOG_DIR}/${job_name}_%j.err"

      # Write individual SLURM script
      cat > "$script_path" <<EOF
#!/usr/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_out}
#SBATCH --error=${log_err}
#SBATCH --time=16:00:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=32GB

module load python/3.9.0
source /home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/stabl_uni_clean/bin/activate

python run_regression_cv.py \\
    --features_path "${features_path}" \\
    --results_dir "${result_dir}" \\
    --artificial_type "${artificial}" \\
    --model_chosen "${model}"
EOF

      # Submit the job
      sbatch "$script_path"

    done
  done
done
