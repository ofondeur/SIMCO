#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import anndata as ad
import fcsparser
import sys
from tqdm import tqdm

# ===== User‐defined directories =====
RAW_DIR = "../cellot_pheno/cells_combined/raw_data/ool"
OUTPUT_DIR = "../datasets/ool_by_patient/unstim_gated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Expected factors =====
path_patients_ina = '../datasets/ptb_concatenated_per_condition_celltype/patients_ina.txt'
with open(path_patients_ina, 'r') as f:
    EXPECTED_PATIENTS = [line.strip() for line in f if line.strip()]
print(EXPECTED_PATIENTS)
EXPECTED_MARKERS = [
    'CD45', 'CD66', 'CD7', 'CD19', 'CD45RA', 'CD11b', 'CD4', 'CD8', 'CD11c',
    'CD123', 'pCREB', 'pSTAT5', 'pp38', 'pSTAT1', 'pSTAT3', 'pS6', 'IkB',
    'pMK2', 'Tbet', 'FoxP3', 'CD16', 'pNFkB', 'pERK', 'pSTAT6', 'CD25',
    'CD3', 'CD62L', 'CCR2', 'HLADR', 'CD14', 'CD56'
]
METAL_MAPPING = {
    "Sm149Di": "pCREB", "Er167Di": "pERK", "Er164Di": "IkB", "Tb159Di": "pMK2",
    "Er166Di": "pNFkB", "Eu151Di": "pp38", "Gd155Di": "pS6", "Eu153Di": "pSTAT1",
    "Sm154Di": "pSTAT3", "Nd150Di": "pSTAT5", "Er168Di": "pSTAT6", "Yb174Di": "HLADR",
    "Tm169Di": "CD25", "Sm152Di": "PD1", "Gd156Di": "CD44", "Gd157Di": "CD36",
    "Gd158Di": "PDL1", "Dy163Di": "GLUT1", "Er170Di": "pPLCg", "Yb171Di": "pSTAT4",
    "La139Di": "CD66", "In115Di": "CD45", "Pr141Di": "CD7", "Nd142Di": "CD19",
    "Nd143Di": "CD45RA", "Sm144Di": "CD11b", "Nd145Di": "CD4", "Nd146Di": "CD8",
    "Sm147Di": "CD11c", "Sm148Di": "CD123", "Gd160Di": "Tbet", "Dy162Di": "FoxP3",
    "Ho165Di": "CD16", "Yb172Di": "CD62L", "Yb173Di": "CCR2", "Lu175Di": "CD14",
    "Lu176Di": "CD56", "Pt198Di": "CD3"
}
EXPECTED_MARKERS = set(EXPECTED_MARKERS)


# ===== Helper functions =====
def normalize_marker(raw_marker):
    marker = str(raw_marker).strip()
    if marker in METAL_MAPPING:
        marker = METAL_MAPPING[marker]
    marker = marker.strip()
    if marker.upper() in EXPECTED_MARKERS:
        return marker.upper()
    raise AssertionError(f"Unrecognized marker: {raw_marker}")

def arcsinh_transform(X, cofactor=5):
    return X.apply(lambda x: np.arcsinh(pd.to_numeric(x, errors='coerce') / cofactor))

def process_fcs_file_simplified(file_path, patient, cell_type, stim):
    meta, data = fcsparser.parse(file_path, reformat_meta=True)
    new_cols = {}
    for col in data.columns:
        try:
            new_name = normalize_marker(col)
            new_cols[col] = new_name
        except AssertionError:
            continue

    if not new_cols:
        print(f"[WARN] No expected markers found in {file_path}")
        return None

    df_subset = data[list(new_cols.keys())].rename(columns=new_cols)
    df_trans = arcsinh_transform(df_subset)

    obs = pd.DataFrame({
        "patient": [patient] * len(df_trans),
        "stim": [stim] * len(df_trans),
        "cell_type": [cell_type] * len(df_trans),
    })

    return ad.AnnData(X=df_trans.values, obs=obs, var=pd.DataFrame(index=df_trans.columns))

# ===== Main processing =====
all_files = [f for f in os.listdir(RAW_DIR) if f.endswith("Unstim.fcs")]
patient_data = {}

for fname in tqdm(all_files, desc="Processing FCS files"):
    try:
        parts = fname.replace(".fcs", "").split("_")
        if len(parts) < 6:
            print(f"[SKIP] Unexpected filename format: {fname}")
            continue
        patient_id = str(int(parts[0]))+'_'+parts[1]
        cell_type = parts[4]
        stim = parts[5]
        if stim != "Unstim":
            continue
        if patient_id not in EXPECTED_PATIENTS:
            print(f"[SKIP] Unknown patient: {patient_id}")
            continue

        file_path = os.path.join(RAW_DIR, fname)
        adata = process_fcs_file_simplified(file_path, patient_id, cell_type, stim)
        if adata is not None:
            patient_data.setdefault(patient_id, []).append(adata)

    except Exception as e:
        print(f"[ERROR] {fname}: {e}")

# ===== Write H5AD per patient =====
for patient, adata_list in patient_data.items():
    combined = ad.concat(adata_list, join='outer', axis=0)
    combined.obs_names_make_unique()
    out_path = os.path.join(OUTPUT_DIR, f"{patient}_Unstim.h5ad")
    combined.write_h5ad(out_path)
    print(f"[SAVED] {out_path} with {combined.n_obs} cells")

print("Finished processing all patients.")
