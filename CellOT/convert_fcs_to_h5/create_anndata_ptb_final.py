#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import anndata as ad
import fcsparser
import sys
from tqdm import tqdm

RAW_DIR = "../cells_combined/raw_data/ptb_final_rawdata"
OUTPUT_DIR = "../datasets/ptb_concatenated_to_batchcorrect"
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
    "CD8Tnaive", "CD8Teff", "CD8Tcells Th1", "CD4negCD8negTcells", "NKT"
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

def normalize_marker(raw_marker):
    marker = str(raw_marker).strip()
    if marker in METAL_MAPPING:
        marker = METAL_MAPPING[marker]
    marker = marker.strip()
    if marker not in EXPECTED_MARKERS:
        marker = marker.upper()
    if marker not in EXPECTED_MARKERS:
        raise AssertionError(f"Unrecognized marker: {raw_marker} -> {marker}")
    return marker

def arcsinh_transform(X, cofactor=5):
    return X.apply(lambda x: np.arcsinh(pd.to_numeric(x, errors='coerce') / cofactor))

FILENAME_RE = re.compile(
    r"^PTB_(?P<patient>[^_]+)_(?P<drug>[A-Za-z0-9]+)_(?P<stim>[A-Za-z0-9]+)\s*-\s*(?P<cell_type>.+)\.fcs$"
)

def parse_filename(filename):
    m = FILENAME_RE.match(filename)
    if not m:
        raise AssertionError(f"Filename does not match expected pattern: {filename}")
    patient = m.group("patient")
    drug = m.group("drug")
    stim = m.group("stim")
    cell_type = m.group("cell_type").strip()
    return patient, drug, stim, cell_type

def process_fcs_file_simplified(file_path, patient, drug, stim, cell_type):
    meta, data = fcsparser.parse(file_path, reformat_meta=True)

    new_cols = {}
    marker_cols_found = []
    for col in data.columns:
        try:
            new_name = normalize_marker(col)
            if new_name in EXPECTED_MARKERS:
                new_cols[col] = new_name
                marker_cols_found.append(new_name)
        except AssertionError:
            continue

    if not new_cols:
        print(f"Warning: No expected markers found in file {file_path}. Skipping file.")
        return None 
    data_subset = data[list(new_cols.keys())].rename(columns=new_cols)

    data_transformed = arcsinh_transform(data_subset, cofactor=5)

    obs = pd.DataFrame({
        "patient": [patient] * data_transformed.shape[0],
        "stim": [stim] * data_transformed.shape[0],
        "drug": [drug] * data_transformed.shape[0],
        "cell_type": [cell_type] * data_transformed.shape[0],
    })
    adata = ad.AnnData(X=data_transformed.values, obs=obs, var=pd.DataFrame(index=data_transformed.columns))
    return adata

# ===== Main processing =====
def main():
    representative_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".fcs")]
    if not representative_files:
        sys.stderr.write("No FCS files found in RAW_DIR.\n"); sys.exit(1)
    print("Checking representative file...")
    rep_file = os.path.join(RAW_DIR, representative_files[0])
    try:
        patient, drug, stim, cell_type = parse_filename(representative_files[0])
        
        rep_adata = process_fcs_file_simplified(rep_file, patient, drug, stim, cell_type)
        if rep_adata is None: raise AssertionError("Rep file processing returned None")
    except Exception as e:
        sys.stderr.write(f"Error processing representative file:\n{e}\n"); sys.exit(1)
    if not EXPECTED_MARKERS.issubset(set(rep_adata.var_names)):
        missing = EXPECTED_MARKERS - set(rep_adata.var_names)
        print(f"Warning: Representative file missing markers: {missing} (Check source FCS or METAL_MAPPING)")
        
    else:
        print("Representative file check passed (found all expected markers).")

    # List all FCS files
    all_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".fcs")]
    # Expected number calculation: 10 patients * 12 drugs * 8 stims * 28 cell types
    expected_file_count = len(EXPECTED_PATIENTS) * len(EXPECTED_DRUGS) * len(EXPECTED_STIMS) * len(EXPECTED_CELL_TYPES)
    print(f"Found {len(all_files)} FCS files. Expecting {expected_file_count}.")
    if len(all_files) != expected_file_count:
        print(f"Warning: File count mismatch.")
    group_dict = {}
    processed_combinations = set()

    for filename in tqdm(all_files, desc="Processing FCS files"):
        try:
            patient, drug, stim, cell_type = parse_filename(filename)
            # Validate against expected factors
            if patient not in EXPECTED_PATIENTS or \
               drug not in EXPECTED_DRUGS or \
               stim not in EXPECTED_STIMS or \
               cell_type not in EXPECTED_CELL_TYPES:
                print(f"Warning: Skipping file {filename} due to unexpected factor(s).")
                continue

            processed_combinations.add((patient, drug, stim, cell_type))
            file_path = os.path.join(RAW_DIR, filename)
            # Use the simplified processor
            adata = process_fcs_file_simplified(file_path, patient, drug, stim, cell_type)

            if adata is not None:
                group_drug = drug  # group_drug = "DMSO" if drug.startswith("DMSO") else drug (if concat all the DMSo together)
                key = (cell_type,(cell_type,patient[:2]))
                group_dict.setdefault(key, []).append(adata)

        except AssertionError as e:
            print(f"Warning: Skipping file {filename} - {e}")
            continue
        except Exception as e:
             print(f"Warning: Error processing file {filename} - {e}")
             continue

    expected_combinations = set((p, dr, s, ct) for p in EXPECTED_PATIENTS for dr in EXPECTED_DRUGS for s in EXPECTED_STIMS for ct in EXPECTED_CELL_TYPES)
    missing = expected_combinations - processed_combinations
    if missing:
        print(f"\nWarning: {len(missing)} expected combinations seem to be missing or were skipped:")
        
    print("\nConcatenating AnnData objects per group (NO deduplication)...")
    for (cell_type,condition),adata_list in tqdm(group_dict.items(), desc="Concatenating groups"):
        if not adata_list:
            print(f"Warning: No valid AnnData objects found for group {cell_type} and {condition}. Skipping.")
            continue

        # Concatenate along cells, keeping all rows (duplicates included)
        combined_final = ad.concat(adata_list, join='outer', axis=0, label=f"{cell_type}_{condition}")
        print(f"Group {cell_type} and {condition}: Concatenated {len(adata_list)} files -> {combined_final.n_obs} cells.")

        print("Skipped deduplication step intentionally.")

        out_filename = f"{cell_type}_{condition}.h5ad"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        try:
            combined_final.write_h5ad(out_path)
            print(f"Saved simplified AnnData for {cell_type} and {condition} with {combined_final.n_obs} cells to {out_path}")
        except Exception as e:
            print(f"Error saving AnnData for {cell_type} and {condition}: {e}")

    # Write patient and feature lists
    print("\nWriting patient and feature lists...")
    all_patients_in_cohort = sorted(EXPECTED_PATIENTS)
    patients_txt = os.path.join(OUTPUT_DIR, "patients.txt")
    with open(patients_txt, "w") as f: f.write("\n".join(all_patients_in_cohort))

    features_txt = os.path.join(OUTPUT_DIR, "features.txt")
    with open(features_txt, "w") as f: f.write("\n".join(sorted(EXPECTED_MARKERS)))
    print(f"Created {os.path.basename(patients_txt)} and {os.path.basename(features_txt)} in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()