#!/usr/bin/env python3
import os
import pandas as pd
"""
Script to generate a job list for cell-wise training based on a knowledge table.
"""

KNOWLEDGE_TABLE_PATH = "../cross_validation/PenMatrix_HV.csv"
JOB_LIST_OUTPUT_PATH = "../datasets/ptb_concatenated_per_condition_celltype/valid_jobs2.txt"
BASE_DIR_FOR_OUTPUT = "../datasets/ptb_concatenated_per_condition_celltype"

os.makedirs(BASE_DIR_FOR_OUTPUT, exist_ok=True)

# ===== Sanitization Function =====
def sanitize_name(name):
    """Sanitizes names for use in paths/filenames."""
    name = str(name) # Ensure string
    name = name.replace(' ', '_')
    name = name.replace('/', '-')
    name = name.replace('+', 'pos')
    name = name.replace('-', 'neg')
    return name

print(f"Reading knowledge table: {KNOWLEDGE_TABLE_PATH}")
try:
    knowledge_df = pd.read_csv(KNOWLEDGE_TABLE_PATH)
except FileNotFoundError:
    print(f"ERROR: Knowledge table not found at {KNOWLEDGE_TABLE_PATH}")
    exit(1)
except Exception as e:
    print(f"ERROR: Failed to read knowledge table: {e}")
    exit(1)

required_cols = ['stim', 'population', 'include_marker']
if not all(col in knowledge_df.columns for col in required_cols):
    print(f"ERROR: Knowledge table missing required columns ({required_cols}). Found: {knowledge_df.columns.tolist()}")
    exit(1)

valid_combos = knowledge_df[knowledge_df['include_marker'] == True][['stim', 'population']].drop_duplicates()

# Convert to list of tuples (stim, original_population)
job_list_raw = sorted(list(valid_combos.itertuples(index=False, name=None)))

num_jobs = len(job_list_raw)

if num_jobs == 0:
    print("WARNING: No valid (stim, cell_type) combinations found with include_marker=True. No jobs to run.")
else:
    print(f"Found {num_jobs} valid (stim, cell_type) combinations to train.")
    # Write jobs to file: stim, sanitized_celltype, original_celltype
    with open(JOB_LIST_OUTPUT_PATH, 'w') as f:
        for stim, original_cell_type in job_list_raw:
            sanitized_cell_type = sanitize_name(original_cell_type)
            f.write(f"{stim},{sanitized_cell_type},{original_cell_type}\n")
    print(f"Job list written to: {JOB_LIST_OUTPUT_PATH}")

print("\n=====================================================")
print("Set the following in thd SLURM script:")
print(f"#SBATCH --array=0-{num_jobs-1}")
print("=====================================================")