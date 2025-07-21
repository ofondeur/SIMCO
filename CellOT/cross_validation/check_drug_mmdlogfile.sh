#!/bin/bash

MODEL="drug_OG13"
drug_used='CHT'
CV_NAME="diffIO_med_HVPV"
RESULTS_BASE="/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_unstim/cross_validation_${MODEL}/${drug_used}"
JOB_LIST="/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/valid_jobs.txt"
MISSING_INDICES_FILE="missing_jobs_${drug_used}_unstim.txt"

missing_any=false
rm -f "$MISSING_INDICES_FILE"\

i=0
while IFS=',' read -r stim sanitized_celltype original_celltype; do
    #((i++))
    OUTDIR="${RESULTS_BASE}/${stim}/${sanitized_celltype}/model-${stim}_${sanitized_celltype}"
    TEST_FILE="${OUTDIR}/cache/mmd_log.csv"

    if [ ! -f "$TEST_FILE" ]; then
        echo "[MISSING] ${stim}/${sanitized_celltype} → ${TEST_FILE} absent"
        echo "$i" >> "$MISSING_INDICES_FILE"
        missing_any=true
    fi
    ((i++))
done < "$JOB_LIST"

if [ "$missing_any" = false ]; then
    echo "All models are fully trained."
else
    echo "A few models are not fully trained, index is saved under $MISSING_INDICES_FILE"
fi

