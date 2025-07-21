#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import anndata as ad

############################
# Mapping dictionaries & normalization functions
############################

CELL_TYPE_MAPPING = {
    "granulocytes": "Granulocytes",
    "granulocytes (cd45-cd66+)": "Granulocytes",
    "neutrophils": "Granulocytes",
    "bcells": "Bcells",
    "b-cells (cd19+cd3-)": "Bcells",
    "b-cells": "Bcells",
    "ncmc": "ncMCs",
    "non-classical monocytes (cd14-cd16+)": "ncMCs",
    "ncmcs": "ncMCs",
    "cmc": "cMCs",
    "classical monocytes (cd14+cd16-)": "cMCs",
    "cmcs": "cMCs",
    "mdsc": "MDSCs",
    "mdscs (lin-cd11b-cd14+hladrlo)": "MDSCs",
    "mdscs": "MDSCs",
    "mdc": "mDCs",
    "mdcs (cd11c+hladr+)": "mDCs",
    "mdcs": "mDCs",
    "pdc": "pDCs",
    "pdcs(cd123+hladr+)": "pDCs",
    "pdcs": "pDCs",
    "intmc": "intMCs",
    "intermediate monocytes (cd14+cd16+)": "intMCs",
    "intmcs": "intMCs",
    "cd56brightcd16negnk": "CD56hiCD16negNK",
    "cd56+cd16- nk cells": "CD56hiCD16negNK",
    "cd16- cd56+ nk": "CD56hiCD16negNK",
    "cd56dimcd16posnk": "CD56loCD16posNK",
    "cd56locd16+nk cells": "CD56loCD16posNK",
    "cd16+ cd56lo nk": "CD56loCD16posNK",
    "nk": "NKcells",
    "nk cells (cd7+)": "NKcells",
    "cd4tcells": "CD4Tcells",
    "cd4 t-cells": "CD4Tcells",
    "cd4 tcells": "CD4Tcells",
    "tregs": "Tregs",
    "treg": "Tregs",
    "tregs (cd25+foxp3+)": "Tregs",
    "cd8tcells": "CD8Tcells",
    "cd8 t-cells": "CD8Tcells",
    "cd8 tcells": "CD8Tcells",
    "cd4negcd8neg": "CD4negCD8negTcells",
    "cd8-cd4- t-cells": "CD4negCD8negTcells",
    "cd4- cd8- t-cells": "CD4negCD8negTcells",
    'surge_prehab_controls - gated populations/controls_1_surge_3_hcff_bl_unstim -cd56dimcd16posnk.fcs': "CD56loCD16posNK"
}

def normalize_cell_type(cell):
    key = cell.strip().lower()
    if key not in CELL_TYPE_MAPPING:
        raise AssertionError(f"Unrecognized cell type: {cell}")
    return CELL_TYPE_MAPPING[key]

STIM_MAPPING = {
    "unstim": "Unstim",
    "tnfa": "TNFa",
    "lps": "LPS",
    "p. gingivalis": "LPS",
    "ifna": "IFNa"
}

def normalize_stim(stim):
    key = stim.strip().lower()
    if key not in STIM_MAPPING:
        raise AssertionError(f"Unrecognized stimulation: {stim}")
    return STIM_MAPPING[key]

def normalize_marker(raw_marker):
    marker = raw_marker.strip().lower()
    # Remove leading "p" characters (except for "pp38")
    while marker.startswith("p"):
        marker = marker[1:]
    if marker == "creb":
        return "pCREB"
    elif marker in ["erk", "erk12"]:
        return "pERK"
    elif marker == "ikb":
        return "IkB"
    elif marker in ["mk2", "mapkapk2"]:
        return "pMK2"
    elif marker == "nfkb":
        return "pNFkB"
    elif marker in ["38", "p38"]:
        return "pp38"
    elif marker == "s6":
        return "pS6"
    elif marker == "stat1":
        return "pSTAT1"
    elif marker == "stat3":
        return "pSTAT3"
    elif marker == "stat5":
        return "pSTAT5"
    elif marker == "stat6":
        return "pSTAT6"
    elif marker == "hladr":
        return "HLADR"
    elif marker == "cd25":
        return "CD25"
    else:
        raise AssertionError(f"Unrecognized marker: {raw_marker}")

############################
# Global constants
############################

# The 13 canonical markers
MARKERS = ["pCREB", "pERK", "IkB", "pMK2", "pNFkB", "pp38", "pS6",
           "pSTAT1", "pSTAT3", "pSTAT5", "pSTAT6", "HLADR", "CD25"]

# List of 15 cell types (exactly as used in your prediction script)
CELL_TYPES = [
    'Granulocytes_(CD45-CD66+)',
    'B-Cells_(CD19+CD3-)',
    'Classical_Monocytes_(CD14+CD16-)',
    'MDSCs_(lin-CD11b-CD14+HLADRlo)',
    'mDCs_(CD11c+HLADR+)',
    'pDCs(CD123+HLADR+)',
    'Intermediate_Monocytes_(CD14+CD16+)',
    'Non-classical_Monocytes_(CD14-CD16+)',
    'CD56+CD16-_NK_Cells',
    'CD56loCD16+NK_Cells',
    'NK_Cells_(CD7+)',
    'CD4_T-Cells',
    'Tregs_(CD25+FoxP3+)',
    'CD8_T-Cells',
    'CD8-CD4-_T-Cells'
]

############################
# Helper functions for loading & processing AnnData files
############################

def load_anndata_to_df(anndata_path):
    """
    Load an AnnData file and return a DataFrame containing expression values (for MARKERS)
    plus all obs columns. Marker, cell type and stim names are normalized.
    """
    try:
        adata = ad.read_h5ad(anndata_path)
    except Exception as e:
        raise AssertionError(f"Error reading file {anndata_path}: {e}")
    df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs.index)
    # Rename marker columns if they can be normalized (only keep canonical markers)
    new_cols = {}
    for col in df.columns:
        try:
            new_name = normalize_marker(col.split("_")[1])
            if new_name in MARKERS:
                new_cols[col] = new_name
        except Exception as e:
            continue
    df = df.rename(columns=new_cols)
    # Append obs columns
    for col in adata.obs.columns:
        df[col] = adata.obs[col]
    # Normalize cell type if present
    if "cell_type" in df.columns:
        print(anndata_path)
        df["cell_type"] = df["cell_type"].apply(normalize_cell_type)
    # Normalize stim if present
    if "stim" in df.columns:
        
        df["stim"] = df["stim"].apply(normalize_stim)
    return df

def compute_group_medians(df, markers=MARKERS):
    """
    Melt the DataFrame so each row is one marker value and then compute the median
    per combination of patient, cell_type, and marker.
    """
    present_markers = [m for m in markers if m in df.columns]
    if len(present_markers) != len(markers):
        missing = set(markers) - set(present_markers)
        raise AssertionError(f"Missing marker columns: {missing} in file with columns {list(df.columns)}")
    # Identify non-marker columns to use as id_vars
    id_vars = [col for col in df.columns if col not in markers]
    df_melt = df.melt(id_vars=id_vars, value_vars=markers, var_name="marker", value_name="value")
    medians = df_melt.groupby(["patient", "cell_type", "marker"])["value"].median().reset_index()
    medians = medians.rename(columns={"value": "median", "patient": "sampleID"})
    return medians
    
def load_and_compute_medians(anndata_path):
    df = load_anndata_to_df(anndata_path)
    medians = compute_group_medians(df)
    return medians

############################
# Main processing
############################

def main():
    out_dir = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/cellwise/results_training_sinplecorr"
    os.makedirs(out_dir, exist_ok=True)

    cohorts_and_stims = {
        "surge": ["LPS"]
    }

    for cohort, stims in cohorts_and_stims.items():
        pred_rows = []
        baseline_store = {}

        for stim in stims:
            for cell in CELL_TYPES:
                if stim == "LPS":
                    stim_folder = "P._gingivalis"
                else:
                    stim_folder = stim
                pred_path = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/perio_on_{cohort}_train_corrected_simple/{stim_folder}_{cell}/model-cellot/pred_{cohort}_simplecorrected_test.h5ad"
                if not os.path.exists(pred_path):
                    raise AssertionError(f"Missing prediction file: {pred_path}")

                df_all = load_anndata_to_df(pred_path)

                # Filtrer les lignes baseline (true unstim corrigé)
                df_baseline = df_all[(df_all["state"] == "true_corrected") & (df_all["drug"] == "Unstim")]
                baseline_medians = compute_group_medians(df_baseline)
                baseline_medians["stim"] = "Unstim"

                if cell in baseline_store:
                    merged_base = pd.merge(baseline_store[cell], baseline_medians, on=["sampleID", "cell_type", "marker"], suffixes=("_old", "_new"))
                    if not np.allclose(merged_base["median_old"], merged_base["median_new"], atol=1e-12):
                        raise AssertionError(f"Inconsistent baseline medians for cell type '{cell}'")
                else:
                    baseline_store[cell] = baseline_medians

                # Filtrer les lignes prédictions
                df_pred = df_all[df_all["state"] == "predicted"]
                pred_medians = compute_group_medians(df_pred)
                pred_medians["stim"] = stim
                patients_presents=baseline_medians['median'].notna()
                pred_medians = pred_medians[patients_presents]
                baseline_medians = baseline_medians[patients_presents]
                # Fusionner pour faire la soustraction
                merged = pd.merge(pred_medians, baseline_medians, on=["sampleID", "cell_type", "marker"], suffixes=("_stim", "_baseline"))
                if merged.empty:
                    raise AssertionError(f"Merge failed for {cohort}, {stim}, {cell}")
                if not np.allclose(merged["median_baseline"], baseline_medians["median"], atol=1e-12):
                    raise AssertionError(f"Mismatch in baseline medians for {cohort}, {stim}, {cell}")

                merged["median_diff"] = merged["median_stim"] - merged["median_baseline"]
                pred_final = merged[["sampleID", "cell_type", "marker"]].copy()
                pred_final["stim"] = stim
                pred_final["median"] = merged["median_diff"]
                pred_rows.append(pred_final)

        baseline_all = pd.concat(list(baseline_store.values()), ignore_index=True)
        baseline_all = baseline_all.drop_duplicates(subset=["sampleID", "cell_type", "marker", "stim"])

        if pred_rows:
            pred_all = pd.concat(pred_rows, ignore_index=True)
        else:
            pred_all = pd.DataFrame(columns=["sampleID", "cell_type", "marker", "stim", "median"])

        final_df = pd.concat([baseline_all, pred_all], ignore_index=True)
        final_df = final_df.rename(columns={"cell_type": "population"})
        final_df = final_df[["sampleID", "population", "marker", "stim", "median"]]

        out_path = os.path.join(out_dir, f"{cohort}_{stim}_predicted_transformed.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved results for cohort '{cohort}', stim '{stim}' to {out_path}")


if __name__ == "__main__":
    main()