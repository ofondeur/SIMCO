#!/bin/bash
model='drug_OG13'

LOG_DIR="/home/groups/gbrice/ptb-drugscreen/ot/cellot/logs_drugs/drug_OG13"

echo "📋 Analyse des fichiers de log dans $LOG_DIR..."

found_any=false

for logfile in "$LOG_DIR"/*; do
    if grep -qE "ERROR|WARNING|CANCELLED|Killed|oom_kill" "$logfile"; then
        echo "  Problem in : $logfile"
        grep -E "ERROR|WARNING|CANCELLED|Killed|oom_kill" "$logfile"
        found_any=true
    fi
done

if [ "$found_any" = false ]; then
    echo "No 'ERROR' or'WARNING' found in the logs."
fi
