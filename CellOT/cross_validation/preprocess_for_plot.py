#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import anndata as ad
from collections import defaultdict
import sys


def load_anndata_to_df(anndata_path,drug_used=None):
    """
    Load an AnnData file and return a DataFrame containing expression values (for MARKERS)
    plus all obs columns. Marker, cell type and stim names are normalized.
    """
    try:
        adata = ad.read_h5ad(anndata_path)
    except Exception as e:
        raise AssertionError(f"Error reading file {anndata_path}: {e}")
    df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs.index)
    # Append obs columns
    for col in adata.obs.columns:
        df[col] = adata.obs[col]
    if drug_used:
        df=df[df['drug']==drug_used]
    
    return df


def compute_group_medians(df, markers):
    """
    Melt the DataFrame so each row is one marker value and then compute the median
    per combination of patient, cell_type, and marker.
    """
    markers = [m for m in markers if m in df.columns]
    
    id_vars = [col for col in df.columns if col not in markers]
    df_melt = df.melt(id_vars=id_vars, value_vars=markers, var_name="marker", value_name="value")
    medians = df_melt.groupby(["patient", "cell_type", "marker"])["value"].median().reset_index()
    medians = medians.rename(columns={"value": "median", "patient": "sampleID"})
    return medians

############################
# Main processing
############################
model='original_median_20marks' # choose among 'olivier', 'peter', 'original','all_markers' etc.


FOLD_INFO_FILE = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info_{model}.csv"
fold_info = defaultdict(lambda: defaultdict(dict)) # Structure: fold_info[stim][sanitized_celltype][fold_index] = [test_patient1, ...]
stim_celltype_pairs_in_folds = set() # Keep track of (stim, original_celltype) pairs processed
try:
    with open(FOLD_INFO_FILE, 'r') as f:
        header = f.readline().strip().split(',') # Read header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4: continue # stim, sanitized, original, fold, patient1...
            stim, sanitized_ct, original_ct, fold_idx_str = parts[:4]
            test_patients = parts[4:]
            try:
                fold_idx = int(fold_idx_str)
                fold_info[stim][sanitized_ct][fold_idx] = test_patients
                stim_celltype_pairs_in_folds.add((stim, original_ct))
            except ValueError:
                print(f"[WARN] Invalid fold index '{fold_idx_str}' in line: {line.strip()}")
    print(f"Loaded fold info for {len(fold_info)} stims.")
    print(f"Total (stim, original_celltype) pairs with fold info: {len(stim_celltype_pairs_in_folds)}")
except FileNotFoundError:
    sys.exit(f"FATAL ERROR: Fold info file not found: {FOLD_INFO_FILE}")
except Exception as e:
    sys.exit(f"FATAL ERROR: Failed to read fold info file: {e}")


PTB_ANNDATA_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype"
MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}"
drug_used='DMSO'

def main():
    out_dir = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/cellwise/cross_validation_{model}"

    os.makedirs(out_dir, exist_ok=True)
    cohort='ptb-drugscreen'
    baseline_store = {}
    for stim in fold_info:
        pred_rows = []
        for sanitized_celltype in fold_info[stim]:
            for fold_number, test_patients in fold_info[stim][sanitized_celltype].items():
                
                result_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}2/model-{stim}_{sanitized_celltype}_fold{fold_number}"
                fold_0_path=f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}2/model-{stim}_{sanitized_celltype}_fold0/pred_CV.h5ad"
                pred_path = f"{result_path}/pred_CV.h5ad"
                
                if not os.path.exists(pred_path):
                    if os.path.exists(fold_0_path):
                        pred_path=fold_0_path
                    else:
                        raise AssertionError(f"Missing prediction file: {pred_path}")
                df_all = load_anndata_to_df(pred_path,drug_used)

                # Filtrer les lignes baseline (true unstim corrigé)
                df_baseline = df_all[(df_all["state"] == "true_corrected") & (df_all["stim"] == "Unstim")]
                with open(f'/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/features_evaluation/features_{stim}_{sanitized_celltype}.txt') as f:
                    markers = [line.strip() for line in f if line.strip()]
                baseline_medians = compute_group_medians(df_baseline,markers)
                baseline_medians["stim"] = "Unstim"

                if sanitized_celltype in baseline_store:
                    merged_base = pd.merge(baseline_store[sanitized_celltype], baseline_medians, on=["sampleID", "cell_type", "marker"], suffixes=("_old", "_new"))
                    if not np.allclose(merged_base["median_old"], merged_base["median_new"], atol=1e-12):
                        raise AssertionError(f"Inconsistent baseline medians for cell type '{sanitized_celltype}'")
                else:
                    baseline_store[sanitized_celltype] = baseline_medians


                # Filtrer les lignes prédictions
                df_pred = df_all[(df_all["state"] == "true_corrected") & (df_all["stim"] == stim)]
                
                pred_medians = compute_group_medians(df_pred,markers)
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
        print(baseline_store.keys())
        baseline_all = pd.concat(list(baseline_store.values()), ignore_index=True)
        baseline_all = baseline_all.drop_duplicates(subset=["sampleID", "cell_type", "marker", "stim"])

        if pred_rows:
            pred_all = pd.concat(pred_rows, ignore_index=True)
        else:
            pred_all = pd.DataFrame(columns=["sampleID", "cell_type", "marker", "stim", "median"])

        final_df = pd.concat([baseline_all, pred_all], ignore_index=True)
        final_df = final_df.rename(columns={"cell_type": "population"})
        final_df = final_df[["sampleID", "population", "marker", "stim", "median"]]

        out_path = os.path.join(out_dir, f"{cohort}_{stim}_transformed.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved results for cohort '{cohort}', stim '{stim}' to {out_path}")


if __name__ == "__main__":
    main()