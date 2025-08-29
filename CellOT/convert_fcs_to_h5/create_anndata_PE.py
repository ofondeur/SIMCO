#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import anndata as ad
import fcsparser
import sys
from tqdm import tqdm

RAW_DIR = "CellOT/PE_HAN_PBMC"
OUTPUT_DIR = "CellOT/datasets/PE_concat"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Expected factors =====
EXPECTED_PATIENTS = {"PV01", "PV02", "PV03", "PV04","PV05","PV06", "PV07", "PV08", "PV09","PV10","PV11", "HV04", "HV05", "HV06", "HV07", "HV08","HV09", "HV10", "HV11"}
EXPECTED_DRUGS = {"CHT", "DMSO1", "DMSO2", "DMSO3", "LPZ", "MAP", "MF", "PRA", "RIF", "SA", "SALPZ", "THF"}
EXPECTED_STIMS = {"Unstim", "TNFa", "LPS", "IL246", "IFNa", "GMCSF", "PI", "IL33"}
EXPECTED_CELL_TYPES = {
    "Granulocytes", "Bcells", "cMCs", "MDSCs", "mDCs", "pDCs", "intMCs", "ncMCs",
    "CD56hiCD16negNK", "CD56loCD16posNK", "NK cells CD11c-", "NK cells CD11c+",
    "CD4Tnaive", "CD4Teff", "CD4Tcm", "CD4Tcm CCR2+", "CD4Tem", "CD4Tem CCR2+",
    "CD4Tregs", "CD8Tcm", "CD8Tcm CCR2+", "CD8Tem", "CD8Tem CCR2+",
    "CD8Tnaive", "CD8Teff", "CD8Tcells Th1", "CD4negCD8negTcells", "NKT","CD4Tmemory","CD8Tmemory","Tregs"
}

# ===== Expected markers =====
FUNCTIONAL_MARKERS = [
    "pCREB", "pERK", "IkB", "pMK2", "pNFkB", "pp38", "pS6",
    "pSTAT1", "pSTAT3", "pSTAT5", "pSTAT6", "HLADR", "CD25",
    "GLUT1", "PD1", "PDL1", "CD44", "CD36", "pPLCg", "pSTAT4"
]
PHENOTYPIC_MARKERS = [
    "CD66", "CD45", "CD7", "CD19", "CD45RA", "CD11b", "CD4",
    "CD8", "CD11c", "CD123", "Tbet", "FoxP3", "CD16", "CD62L",
    "CCR2", "CD14", "CD56", "CD3"
]
EXPECTED_MARKERS = set(FUNCTIONAL_MARKERS + PHENOTYPIC_MARKERS)

METAL_MAPPING = {
    "149Sm_CREB": "pCREB",
    "171Yb_pERK": "pERK",
    "164Dy_IkB": "IkB",
    "159Tb_MAPKAPK2": "pMK2",
    "166Er_NFkB": "pNFkB",
    "151Eu_p38": "pp38",
    "155Gd_S6": "pS6",
    "153Eu_STAT1": "pSTAT1",
    "154Sm_STAT3": "pSTAT3",
    "150Nd_STAT5": "pSTAT5",
    "168Er_pSTAT6": "pSTAT6",
    "174Yb_HLADR": "HLADR",
    "169Tm_CD25": "CD25",
    "162Dy_FoxP3": "FoxP3",
    "160Gd_Tbet": "Tbet",
    "139La_CD66": "CD66",
    "115In_CD45": "CD45",
    "141Pr_CD7": "CD7",
    "142Nd_CD19": "CD19",
    "143Nd_CD45RA": "CD45RA",
    "144Nd_CD11b": "CD11b",
    "145Nd_CD4": "CD4",
    "146Nd_CD8a": "CD8",
    "147Sm_CD11c": "CD11c",
    "148Nd_CD123": "CD123",
    "165Ho_CD16": "CD16",
    "175Lu_CD14": "CD14",
    "176Yb_CD56": "CD56",
    "170Er_CD3": "CD3",
}


def normalize_marker(raw_marker):
    marker = str(raw_marker).strip()
    if marker in METAL_MAPPING:
        marker = METAL_MAPPING[marker]
    marker = marker.strip()
    if marker not in EXPECTED_MARKERS:
        marker = marker.upper()
        raise AssertionError(f"Unrecognized marker: {raw_marker} -> {marker}")
    return marker

def arcsinh_transform(X, cofactor=5):
    return X.apply(lambda x: np.arcsinh(pd.to_numeric(x, errors='coerce') / cofactor))

FILENAME_RE = re.compile(
    r"^(?P<patient>\d+)_(?P<GA>\d+)_(?P<population>[^_]+)_(?P<group>Ctrl|PreE)\.fcs$"
)

def parse_filename(filename):
    m = FILENAME_RE.match(filename)
    if not m:
        raise AssertionError(f"Filename does not match expected pattern: {filename}")
    patient = m.group("patient")
    GA = m.group("GA")
    cell_type = m.group("population").strip()
    group = m.group("group").strip()
    return patient, GA, cell_type, group

def process_fcs_file_simplified(file_path, patient, GA, group, cell_type):
    meta, data = fcsparser.parse(file_path, reformat_meta=True)

    new_cols = {}
    marker_cols_found = []
    for col in data.columns:
        try:
            new_name = normalize_marker(col)
            if new_name in EXPECTED_MARKERS:
                new_cols[col] = new_name
                marker_cols_found.append(new_name)
            else:
                print(f"Warning: Marker {new_name} not in expected markers, skipping.")
        except AssertionError:
            continue
    if not new_cols:
        print(f"Warning: No expected markers found in file {file_path}. Skipping file.")
        return None 
    data_subset = data[list(new_cols.keys())].rename(columns=new_cols)

    data_transformed = arcsinh_transform(data_subset, cofactor=5)

    obs = pd.DataFrame({
        "patient": [f"{patient}_{GA}"] * data_transformed.shape[0],
        "group": [group] * data_transformed.shape[0],
        "cell_type": [cell_type] * data_transformed.shape[0],
    })
    adata = ad.AnnData(X=data_transformed.values, obs=obs, var=pd.DataFrame(index=data_transformed.columns))
    return adata

# ===== Main processing =====
def main():
    bad=[]
    representative_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".fcs")]
    if not representative_files:
        sys.stderr.write("No FCS files found in RAW_DIR.\n"); sys.exit(1)
    print("Checking representative file...")
    rep_file = os.path.join(RAW_DIR, representative_files[0])
    try:
        patient, GA, cell_type, group = parse_filename(representative_files[0])

        rep_adata = process_fcs_file_simplified(rep_file, patient, GA, group, cell_type)
        if rep_adata is None: raise AssertionError("Rep file processing returned None")
    except Exception as e:
        sys.stderr.write(f"Error processing representative file:\n{e}\n"); sys.exit(1)
    if not EXPECTED_MARKERS.issubset(set(rep_adata.var_names)):
        missing = EXPECTED_MARKERS - set(rep_adata.var_names)
        print(f"Warning: Representative file missing markers: {missing} (Check source FCS or METAL_MAPPING)")
        
    else:
        print("Representative file check passed (found all expected markers).")
    all_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".fcs")]
    processed_adatas = []
    processed_combinations = set()

    for filename in tqdm(all_files, desc="Processing FCS files"):
        try:
            patient, GA, cell_type, group = parse_filename(filename)
            # Validate against expected factors
            if group not in ['Ctrl','PreE']:
                print(f"Warning: Skipping file {filename} due to unexpected factor(s): {group}")
                continue
            if cell_type not in EXPECTED_CELL_TYPES:
                bad.append(f"{cell_type}'")
                raise ValueError(f"Unexpected cell type '{cell_type}' in file {filename}")

            processed_combinations.add((patient, GA, group, cell_type))
            file_path = os.path.join(RAW_DIR, filename)

            adata = process_fcs_file_simplified(file_path, patient, GA, group, cell_type)

            if adata is not None:
                processed_adatas.append(adata)

        except AssertionError as e:
            print(f"Warning: Skipping file {filename} - {e}")
            continue
        except Exception as e:
            print(f"Warning: Error processing file {filename} - {e}")
            continue

    print("\nConcatenating ALL AnnData objects together (NO deduplication)...")
    if processed_adatas:
        combined_final = ad.concat(processed_adatas, join='outer', axis=0, label="batch")
        print(f"Concatenated {len(processed_adatas)} files -> {combined_final.n_obs} cells.")

        out_filename = "all_data_preeclampsia_tregfixed.h5ad"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        all_markers=combined_final.var_names.tolist()
        print(f"Markers in final data: {all_markers}")
        try:
            combined_final.write_h5ad(out_path)
            print(f"Saved global AnnData with {combined_final.n_obs} cells to {out_path}")
        except Exception as e:
            print(f"Error saving global AnnData: {e}")
    else:
        print("No valid AnnData objects were processed. Nothing to concatenate.")
    print("bad cells:", bad)
    print(f"populations : {combined_final.obs['cell_type'].unique()}")
    # Save all unique patients in combined_final to a txt file
    if processed_adatas:
        unique_patients = combined_final.obs['patient'].unique()
        patients_txt_path = os.path.join(OUTPUT_DIR, "patients_PeE.txt")
        with open(patients_txt_path, "w") as f:
            for patient in unique_patients:
                f.write(f"{patient}\n")
        print(f"Saved list of patients to {patients_txt_path}")

if __name__ == "__main__":
    main()