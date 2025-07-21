#!/bin/bash

MODEL="drug_OG13"
drug_used='PRA'
RESULTS_BASE="/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_drug/cross_validation_${MODEL}/${drug_used}"
JOB_LIST="/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/valid_jobs.txt"

output_ids="missing_job_ids_${drug_used}.txt"
> "$output_ids"

i=0
while IFS=',' read -r stim sanitized_celltype original_celltype; do
    OUTDIR="${RESULTS_BASE}/${stim}/${sanitized_celltype}/model-${stim}_${sanitized_celltype}"
    TEST_FILE="${OUTDIR}/cache/mmd_log.csv"

    if [ ! -f "$TEST_FILE" ]; then
        echo "[MISSING] ${stim}/${sanitized_celltype}"
        echo "$i" >> "$output_ids"
    fi
    i=$((i+1))
done < "$JOB_LIST"

echo "✅ Liste des jobs manquants enregistrée dans $output_ids"
