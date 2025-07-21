#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import anndata as ad
from collections import defaultdict
import sys
import gc
from cellot.data.cell import read_list
from sklearn.metrics.pairwise import rbf_kernel

def compute_univariate_mmd(x, y, gamma=1.0):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xx = rbf_kernel(x, x, gamma=gamma).mean()
    yy = rbf_kernel(y, y, gamma=gamma).mean()
    xy = rbf_kernel(x, y, gamma=gamma).mean()
    return xx + yy - 2 * xy
    
def compute_univariate_mmd_batched(x, y, gamma=1.0, batch_size=500):
    """
    Approximate univariate MMD between x and y by batching to avoid memory issues.
    """
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    def batch_kernel_mean(a, b):
        total = 0.0
        count = 0
        for i in range(0, len(a), batch_size):
            a_batch = a[i:i + batch_size]
            for j in range(0, len(b), batch_size):
                b_batch = b[j:j + batch_size]
                k = rbf_kernel(a_batch, b_batch, gamma=gamma)
                total += k.sum()
                count += k.size
        return total / count if count > 0 else 0.0

    xx = batch_kernel_mean(x, x)
    yy = batch_kernel_mean(y, y)
    xy = batch_kernel_mean(x, y)
    return xx + yy - 2 * xy
    
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
    
    #id_vars = [col for col in df.columns if col not in markers]
    id_vars=['patient','stim','cell_type']
    df_melt = df.melt(id_vars=id_vars, value_vars=markers, var_name="marker", value_name="value")
    medians = df_melt.groupby(["patient", "cell_type", "marker"])["value"].median().reset_index()
    medians = medians.rename(columns={"value": "median", "patient": "sampleID"})
    return medians


############################
# Main processing
############################

model='shuffled_20marks' # choose among 'olivier', 'peter', 'original','all_markers','original_median_20marks','original_20marks'


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
MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_jakob/cross_validation_{model}"
drug_used='DMSO'

def main():
    out_dir = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/cellwise/cross_validation_{model}"
    os.makedirs(out_dir, exist_ok=True)

    cohort = 'ptb-drugscreen'
    already_written = set()

    for stim in fold_info:
        for sanitized_celltype in fold_info[stim]:
            for fold_number, test_patients in fold_info[stim][sanitized_celltype].items():

                result_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}"
                pred_path = f"{result_path}/pred_CV_{model}.h5ad"
                fold_0_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold0/pred_CV_{model}.h5ad"

                if not os.path.exists(pred_path):
                    print(f"################### {pred_path} does not exist ####################")
                    if os.path.exists(fold_0_path):
                        pred_path = fold_0_path
                    else:
                        raise AssertionError(f"Missing prediction file: {pred_path}")

                df_all = load_anndata_to_df(pred_path, drug_used)

                markers = read_list("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/features.txt")
                marks_to_use = [m for m in markers if m in df_all.columns]

                df_pred = df_all[(df_all["state"] == "predicted") & (df_all["stim"] == stim)]
                df_true = df_all[(df_all["state"] == "true_corrected") & (df_all["stim"] == stim)]

                pred_medians = compute_group_medians(df_pred, marks_to_use)
                pred_medians["stim"] = stim

                mmd_list = []

                for i, row in pred_medians.iterrows():
                    sid, ct, marker = row["sampleID"], row["cell_type"], row["marker"]

                    values_pred = df_pred[
                        (df_pred["patient"] == sid) &
                        (df_pred["cell_type"] == ct)
                    ][marker].to_numpy()

                    values_true = df_true[
                        (df_true["patient"] == sid) &
                        (df_true["cell_type"] == ct)
                    ][marker].to_numpy()

                    values_pred = values_pred.astype(np.float32)
                    values_true = values_true.astype(np.float32)

                    max_samples = int(min(len(values_pred), len(values_true)) / 2)
                    max_samples = min(max_samples, 15000)

                    if len(values_pred) > max_samples:
                        values_pred = np.random.choice(values_pred, max_samples, replace=False)
                    if len(values_true) > max_samples:
                        values_true = np.random.choice(values_true, max_samples, replace=False)

                    mmd_val = compute_univariate_mmd_batched(values_pred, values_true, gamma=1.0, batch_size=1000)
                    mmd_list.append(mmd_val)

                pred_medians["mmd"] = mmd_list
                pred_final = pred_medians.rename(columns={"cell_type": "population"})
                pred_final = pred_final[["sampleID", "population", "marker", "stim", "median", "mmd"]]

                out_path = os.path.join(out_dir, f"{cohort}_{stim}_predicted_transformed.csv")
                write_header = stim not in already_written and not os.path.exists(out_path)
                pred_final.to_csv(out_path, mode='a', index=False, header=write_header)
                already_written.add(stim)

                del df_all, df_pred, df_true, pred_medians, pred_final
                gc.collect()

        print(f"Finished all folds for cohort '{cohort}', stim '{stim}'")




if __name__ == "__main__":
    main()