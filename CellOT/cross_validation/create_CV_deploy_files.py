import csv
from pathlib import Path
"""
# This script generates a CSV file containing fold information for cross-validation in a drug screening dataset.
"""
model='original_20marks'
CV_DIR = Path(f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}")

VALID_JOBS_PATH = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/valid_jobs.txt"

NUM_FOLDS = 4
OUTPUT_CSV = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info_{model}.csv"

rows = []

with open(VALID_JOBS_PATH, "r") as f:
    for line in f:
        stim, sanitized_celltype, original_celltype = line.strip().split(",")

        for fold_idx in range(NUM_FOLDS):
            fold_dir = CV_DIR / stim / f"{sanitized_celltype}" / f"model-{stim}_{sanitized_celltype}_fold{fold_idx}"
            test_file = fold_dir / "test_patients.txt"

            if test_file.exists():
                with open(test_file, "r") as tf:
                    patients = tf.read().strip().split()
                while len(patients) < 2:
                    patients.append("")
    
                row = [stim, sanitized_celltype, original_celltype, fold_idx] + patients
                rows.append(row)
                
            else:
                print(f"[WARN] Missing: {test_file}")


with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["stim", "sanitized_celltype", "original_celltype", "fold_index", "test_patient_1", "test_patient_2"])
    writer.writerows(rows)

print(f"[INFO] CSV written to {OUTPUT_CSV}")
