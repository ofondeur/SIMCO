#!/usr/bin/env python3
import os
import pandas as pd

KNOWLEDGE_TABLE_PATH = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/cross_validation/PenMatrix_HV.csv"
OUTPUT_TXT_PATH = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/job_list_all_drugs.txt"
DRUG_LIST = ['SA', 'RIF', 'SALPZ', 'CHT', 'THF', 'LPZ', 'MAP', 'PRA', 'MF']

def sanitize_name(name):
    name = str(name)
    name = name.replace(' ', '_').replace('/', '-').replace('+', 'pos').replace('-', 'neg')
    return name

try:
    knowledge_df = pd.read_csv(KNOWLEDGE_TABLE_PATH)
except FileNotFoundError:
    print(f"[ERROR] Knowledge table not found: {KNOWLEDGE_TABLE_PATH}")
    exit(1)

if 'population' not in knowledge_df.columns:
    print("[ERROR] Column 'population' not found in knowledge table.")
    exit(1)

unique_celltypes = sorted(knowledge_df['population'].dropna().unique())

with open(OUTPUT_TXT_PATH, 'w') as f:
    for drug in DRUG_LIST:
        for celltype in unique_celltypes:
            sanitized = sanitize_name(celltype)
            f.write(f"{drug},{sanitized},{celltype}\n")

print(f"job list written to: {OUTPUT_TXT_PATH}")
print(f"Total jobs: {len(DRUG_LIST) * len(unique_celltypes)}")
