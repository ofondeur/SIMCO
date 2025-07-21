#!/usr/bin/env python3
import os
import sys
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import anndata as ad
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import traceback

# --- Add CellOT library path ---
# Adjust this path if your cellot library is installed elsewhere or you have a specific source dir
CELLOT_PHENOPATH = '/home/groups/gbrice/ptb-drugscreen/ot/cellot'
if CELLOT_PHENOPATH not in sys.path:
    sys.path.insert(0, CELLOT_PHENOPATH)
print(f"Using CellOT library from: {CELLOT_PHENOPATH}")

try:
    from cellot.utils.helpers import load_config
    from cellot.utils.loaders import load
    from cellot.models.cellot import load_networks
    from cellot.data.cell import AnnDataDataset, read_list
    print("CellOT imports successful.")
except ImportError as e:
    print(f"ERROR: Failed to import CellOT modules from {CELLOT_PHENOPATH}. Check path and installation.")
    print(e)
    sys.exit(1)
    
model='peter' # choose among 'olivier', 'peter', 'original','all_markers' 

PTB_ANNDATA_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype"
MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}"
FOLD_INFO_FILE = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info.csv"
OUTPUT_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/cells_combined/cross_validation_plots_{model}/median_diffs_cv"
PRED_CSV_OUT = os.path.join(OUTPUT_DIR, "ptb_predicted_median_diffs_cellwise.csv")
GT_CSV_OUT = os.path.join(OUTPUT_DIR, "ptb_groundtruth_median_diffs_cellwise.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Constants ────────────────────────────────────────────────────────────
# Functional markers (define the target markers for output CSVs)
FUNCTIONAL_MARKERS = [
    "pCREB", "pERK", "IkB", "pMK2", "pNFkB", "pp38", "pS6",
    "pSTAT1", "pSTAT3", "pSTAT5", "pSTAT6", "HLADR", "CD25"
]
print(f"Targeting {len(FUNCTIONAL_MARKERS)} functional markers for output.")

# All expected stims (used for iteration)
ALL_STIMS = ["TNFa", "LPS", "IL246", "IFNa", "GMCSF", "PI", "IL33"]

# Expected columns in PTB AnnData obs
PTB_OBS_COLS = ['patient', 'cell_type'] # Add others if needed by your data

COFACTOR = 5 # For arcsinh transform
BATCH_SIZE = 512 # For prediction DataLoader
N_FOLDS = 5

# ── Helper Functions ─────────────────────────────────────────────────────
def inverse_arcsinh_transform(Y, cofactor=COFACTOR):
    """Reverses the arcsinh transform."""
    Y_numeric = pd.to_numeric(Y, errors='coerce')
    Y_numeric = Y_numeric.fillna(np.nan)
    return np.sinh(Y_numeric) * cofactor

def arcsinh_transform(X, cofactor=COFACTOR):
    """Applies the arcsinh transform."""
    X_numeric = pd.to_numeric(X, errors='coerce')
    X_numeric = X_numeric.fillna(np.nan)
    return np.arcsinh(X_numeric / cofactor)

def calculate_medians_from_df(df: pd.DataFrame, marker_cols: list, group_cols=["patient", "cell_type","stim"]) -> pd.DataFrame:
    """
    Calculates medians from a DataFrame containing patient, cell_type, and marker values.
    Applies inverse arcsinh -> median -> forward arcsinh.
    """
    if df.empty or not marker_cols:
        # print(f"[DEBUG] Empty DataFrame or no marker columns provided to calculate_medians.")
        return pd.DataFrame()

    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        print(f"[ERROR] Median Calc: Missing grouping columns: {missing_group_cols}")
        return pd.DataFrame()

    available_markers = [m for m in marker_cols if m in df.columns]
    if not available_markers:
         # print(f"[DEBUG] Median Calc: None of the requested marker columns found.")
         return pd.DataFrame()

    # 1. Inverse transform
    df_temp = df[group_cols + available_markers].copy()
    for marker in available_markers:
        df_temp[marker] = inverse_arcsinh_transform(df_temp[marker])

    # 2. Calculate median on the "raw scale"
    try:
        raw_medians_df = (
            df_temp.groupby(group_cols, observed=True, dropna=False)[available_markers]
            .median()
            .reset_index()
        )
    except Exception as e:
        print(f"[ERROR] Median Calc: Failed during groupby/median operation: {e}")
        print(f"DataFrame info:\n{df_temp.info()}")
        return pd.DataFrame()


    # 3. Re-apply forward arcsinh transform
    final_medians_df = raw_medians_df.copy()
    for marker in available_markers:
        final_medians_df[marker] = arcsinh_transform(final_medians_df[marker])

    # Add back columns for any markers that weren't available (filled with NaN)
    for m in marker_cols:
        if m not in final_medians_df.columns:
            final_medians_df[m] = np.nan

    # Ensure original marker column order plus group cols
    final_output_cols = group_cols + marker_cols
    cols_to_return = [col for col in final_output_cols if col in final_medians_df.columns]
    final_medians_df = final_medians_df[cols_to_return]

    return final_medians_df

def load_ptb_anndata(path, label):
    """Loads a PTB AnnData file with basic checks."""
    print(f"Loading PTB {label} data from: {path}")
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    try:
        adata = ad.read_h5ad(path)
        print(f"  Successfully loaded {label}. Shape: {adata.shape}")
        if not adata.obs_names.is_unique: adata.obs_names_make_unique()
        if not adata.var_names.is_unique: adata.var_names_make_unique()

        # Verify required obs columns
        missing_obs = [c for c in PTB_OBS_COLS if c not in adata.obs.columns]
        if missing_obs:
            print(f"[ERROR] {label} AnnData missing required obs columns: {missing_obs}")
            return None

        # Verify functional markers are present
        markers_in_adata = list(adata.var_names)
        missing_func_markers = [m for m in FUNCTIONAL_MARKERS if m not in markers_in_adata]
        if missing_func_markers:
            print(f"[WARN] {label} data is missing some functional markers: {missing_func_markers}")
        return adata
    except Exception as e:
        print(f"[ERROR] Failed to load or process {label} AnnData: {path}")
        print(traceback.format_exc())
        return None


# --- Main Execution ---

# ── 1. Load Baseline (Unstim) PTB Data ───────────────────────────────────
adata_ptb_unstim = load_ptb_anndata(PTB_UNSTIM_H5AD, "Unstim Baseline")
if adata_ptb_unstim is None:
    sys.exit("FATAL ERROR: Could not load baseline PTB Unstim data.")

# ── 2. Calculate Baseline Medians ────────────────────────────────────────
print(f"\n[2/6] Calculating baseline medians for {len(FUNCTIONAL_MARKERS)} functional markers...")
markers_in_baseline = [m for m in FUNCTIONAL_MARKERS if m in adata_ptb_unstim.var_names]
if not markers_in_baseline:
     sys.exit("FATAL ERROR: No functional markers found in baseline data variables.")

baseline_df_for_median = pd.concat([
    adata_ptb_unstim.obs[PTB_OBS_COLS].reset_index(drop=True),
    pd.DataFrame(adata_ptb_unstim[:, markers_in_baseline].X, columns=markers_in_baseline).reset_index(drop=True)
], axis=1)

baseline_medians = calculate_medians_from_df(baseline_df_for_median, markers_in_baseline)

if baseline_medians.empty:
     sys.exit("FATAL ERROR: Baseline median calculation resulted in empty dataframe.")

baseline_patients = sorted(baseline_medians['patient'].unique())
print(f"Calculated baseline medians for {len(baseline_patients)} unique patients.")
print(f"Baseline medians DataFrame shape: {baseline_medians.shape}")

# Build baseline lookup dictionary: {(patient, stim, marker): median_value}
baseline_lookup = {}
for _, row in baseline_medians.iterrows():
    pat, stim = row['patient'], row['stim']
    for marker in markers_in_baseline: # Only store for markers present in baseline
        if marker in row and pd.notna(row[marker]):
             baseline_lookup[(pat, stim, marker)] = row[marker]
print(f"Built baseline lookup dictionary with {len(baseline_lookup):,} entries.")

# ── 3. Load Fold Information ─────────────────────────────────────────────
print(f"\n[3/6] Loading fold information from: {FOLD_INFO_FILE}")
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

# ── 4. Load Ground Truth Stimulated Data (Iteratively) ───────────────────
print("\n[4/6] Pre-loading ground truth stimulated data...")
ground_truth_stim_data = {}
for sanitized_ct in ALL_CELLS:
    gt_path = os.path.join(PTB_ANNDATA_DIR, f"{sanitized_ct}_HV.h5ad")
    adata_gt = load_ptb_anndata(gt_path, f"{sanitized_ct} Ground Truth")
    if adata_gt is not None:
        ground_truth_stim_data[sanitized_ct] = adata_gt
    else:
        print(f"[WARN] Could not load ground truth data for {stim}. Ground truth shifts will not be calculated for this stim.")

# ── 5. Iterate, Predict, Calculate Medians & Deltas ──────────────────────
print("\n[5/6] Processing folds: Predicting and calculating shifts...")
all_predicted_deltas = []
all_ground_truth_deltas = []
models_processed_count = 0
models_skipped_count = 0

# Iterate through stims found in the fold info file
for stim, celltype_data in tqdm(fold_info.items(), desc="Processing celltype_data"):
    print(f"\n--- Stim: {stim} ---")

    # Get ground truth data for this stim if loaded
    adata_gt_stim = ground_truth_stim_data.get(celltype_data)
    if adata_gt_stim is None:
        print(f"  Skipping ground truth calculation for {stim} (data not loaded).")

    # Iterate through cell types for this stim
    for sanitized_celltype, fold_data in tqdm(celltype_data.items(), desc=f"  Cell Types ({stim})", leave=False):

        # Find the original cell type name (needed for filtering)
        # We can infer this from the first fold's entry, assuming consistency
        try:
            example_fold_idx = list(fold_data.keys())[0]
            example_test_patients = fold_data[example_fold_idx]
            # Need to look up original name again - find it in the fold info source if possible?
            # Let's re-read the fold info file slightly differently or assume consistency
            # Quick fix: Use the fold_info structure to find a corresponding line
            original_celltype = None
            with open(FOLD_INFO_FILE, 'r') as f:
                header = f.readline()
                for line in f:
                     parts = line.strip().split(',')
                     if len(parts) >= 4 and parts[0]==stim and parts[1]==sanitized_celltype:
                         original_celltype = parts[2]
                         break
            if original_celltype is None:
                print(f"    [ERROR] Could not determine original cell type for {stim}/{sanitized_celltype}. Skipping.")
                continue
            # print(f"    Processing Cell Type: {original_celltype} (Sanitized: {sanitized_celltype})")
        except Exception as e:
             print(f"    [ERROR] Issue getting cell type info for {stim}/{sanitized_celltype}: {e}. Skipping.")
             continue


        # Iterate through the folds for this stim/celltype
        for fold_idx, test_patients in fold_data.items():
            # print(f"      Processing Fold {fold_idx} (Test Patients: {test_patients})...") # Verbose

            # --- Locate Model ---
            job_name_fold = f"{stim}_{sanitized_celltype}_fold{fold_idx}"
            model_dir = os.path.join(MODEL_BASE_DIR, stim, sanitized_celltype, f"model-{job_name_fold}")

            cfg_path = os.path.join(model_dir, "config.yaml")
            chkpt_path = os.path.join(model_dir, "cache", "model.pt")

            if not os.path.isdir(model_dir) or not os.path.exists(cfg_path) or not os.path.exists(chkpt_path):
                # print(f"      [WARN] Model/config/checkpoint missing for Fold {fold_idx}, skipping: {model_dir}")
                models_skipped_count += 1
                continue

            # --- Load Config & Features ---
            try:
                config = load_config(cfg_path)
                features_path = config.data.get('features')
                if not features_path or not os.path.exists(features_path):
                    print(f"      [WARN] Features file missing for Fold {fold_idx}, skipping.")
                    models_skipped_count += 1
                    continue
                trained_markers = read_list(features_path)
                if not trained_markers:
                    print(f"      [WARN] No features loaded for Fold {fold_idx}, skipping.")
                    models_skipped_count += 1
                    continue
                # print(f"      Loaded {len(trained_markers)} trained features for Fold {fold_idx}.") # Verbose
            except Exception as e:
                print(f"      [ERROR] Failed loading config/features for Fold {fold_idx}: {e}")
                models_skipped_count += 1
                continue

            # --- Prepare Input Data (Baseline for Test Patients/Cell Type) ---
            input_adata_unstim = adata_ptb_unstim[
                (adata_ptb_unstim.obs['patient'].isin(test_patients)) &
                (adata_ptb_unstim.obs['cell_type'] == original_celltype)
            ].copy()

            if input_adata_unstim.n_obs == 0:
                # print(f"      [INFO] No baseline cells found for test patients {test_patients} / cell type {original_celltype}. Skipping fold.")
                continue # Skip prediction and GT calc for this fold if no input cells

            # Filter baseline further by trained markers for prediction input
            markers_in_input = list(input_adata_unstim.var_names)
            markers_to_use_pred = [m for m in trained_markers if m in markers_in_input]
            if not markers_to_use_pred:
                print(f"      [WARN] Fold {fold_idx}: None of the trained markers present in baseline input data for {original_celltype}. Skipping prediction.")
                continue # Cant predict if no overlapping markers

            adata_input_for_pred = input_adata_unstim[:, markers_to_use_pred].copy()
            # print(f"      Prepared prediction input data shape: {adata_input_for_pred.shape}") # Verbose

            # --- Load Model & Predict ---
            try:
                # Load model, passing actual input data to avoid reading config path
                (f, g), _, _, _ = load(
                    config,
                    restore=chkpt_path,
                    data=adata_input_for_pred, # Pass the actual data
                    split_on=[],
                    include_model_kwargs=True,
                )
                g.eval()

                # Prepare DataLoader
                ds = AnnDataDataset(adata_input_for_pred, include_index=False)
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

                # Run predictions
                preds_list = []
                for batch in loader: # No need for tqdm here, usually fast
                    batch = batch.requires_grad_(True)
                    out = g.transport(batch).detach().cpu().numpy()
                    preds_list.append(out)

                if not preds_list: continue # Skip if prediction failed

                X_pred = np.vstack(preds_list)
                assert X_pred.shape == adata_input_for_pred.shape, "Prediction output shape mismatch!"

                # --- Calculate Predicted Medians & Deltas ---
                df_pred_for_median = pd.concat([
                    adata_input_for_pred.obs[PTB_OBS_COLS].reset_index(drop=True),
                    pd.DataFrame(X_pred, columns=markers_to_use_pred).reset_index(drop=True)
                ], axis=1)

                predicted_medians = calculate_medians_from_df(df_pred_for_median, markers_to_use_pred)

                if not predicted_medians.empty:
                    # Melt and merge with baseline
                    pred_melted = predicted_medians.melt(
                        id_vars=["patient", "cell_type"],
                        value_vars=markers_to_use_pred,
                        var_name="marker", value_name="median_pred"
                    )
                    pred_merged = pd.merge(
                        pred_melted,
                        baseline_medians.melt(id_vars=["patient", "cell_type"], value_vars=markers_in_baseline, var_name="marker", value_name="median_base")[["patient", "cell_type", "marker", "median_base"]],
                        on=["patient", "cell_type", "marker"], how="left"
                    )
                    pred_merged["median_diff"] = pred_merged["median_pred"] - pred_merged["median_base"]
                    pred_merged["stim"] = stim # Add stim column
                    # Keep only rows where diff could be calculated and marker is functional
                    pred_deltas = pred_merged.dropna(subset=['median_diff'])
                    pred_deltas = pred_deltas[pred_deltas['marker'].isin(FUNCTIONAL_MARKERS)]
                    all_predicted_deltas.append(pred_deltas[["patient", "cell_type", "marker", "stim", "median_diff"]])
                    # print(f"      Added {len(pred_deltas)} predicted delta rows for Fold {fold_idx}.") # Verbose

                models_processed_count += 1 # Increment only if prediction step succeeds

            except Exception as e:
                print(f"      [ERROR] Failed prediction step for Fold {fold_idx}: {e}")
                print(traceback.format_exc())
                models_skipped_count += 1
                # Continue to ground truth calculation even if prediction fails

            # --- Calculate Ground Truth Medians & Deltas ---
            if adata_gt_stim is not None:
                # Filter GT data for test patients and cell type
                adata_gt_subset = adata_gt_stim[
                    (adata_gt_stim.obs['patient'].isin(test_patients)) &
                    (adata_gt_stim.obs['cell_type'] == original_celltype)
                ].copy()

                if adata_gt_subset.n_obs > 0:
                    # Calculate GT medians for all available functional markers
                    markers_in_gt = [m for m in FUNCTIONAL_MARKERS if m in adata_gt_subset.var_names]
                    if markers_in_gt:
                        df_gt_for_median = pd.concat([
                            adata_gt_subset.obs[PTB_OBS_COLS].reset_index(drop=True),
                            pd.DataFrame(adata_gt_subset[:, markers_in_gt].X, columns=markers_in_gt).reset_index(drop=True)
                        ], axis=1)

                        gt_medians = calculate_medians_from_df(df_gt_for_median, markers_in_gt)

                        if not gt_medians.empty:
                            gt_melted = gt_medians.melt(
                                id_vars=["patient", "cell_type"],
                                value_vars=markers_in_gt,
                                var_name="marker", value_name="median_gt"
                            )
                            gt_merged = pd.merge(
                                gt_melted,
                                baseline_medians.melt(id_vars=["patient", "cell_type"], value_vars=markers_in_baseline, var_name="marker", value_name="median_base")[["patient", "cell_type", "marker", "median_base"]],
                                on=["patient", "cell_type", "marker"], how="left"
                            )
                            gt_merged["median_diff"] = gt_merged["median_gt"] - gt_merged["median_base"]
                            gt_merged["stim"] = stim
                            gt_deltas = gt_merged.dropna(subset=['median_diff'])
                            # Keep only functional markers (already filtered by markers_in_gt)
                            all_ground_truth_deltas.append(gt_deltas[["patient", "cell_type", "marker", "stim", "median_diff"]])
                            # print(f"      Added {len(gt_deltas)} ground truth delta rows for Fold {fold_idx}.") # Verbose
                # else: # No GT cells for this subset
                     # print(f"      [INFO] No ground truth cells found for test patients {test_patients} / cell type {original_celltype}.")


# ── 6. Assemble and Save Final CSVs ──────────────────────────────────────
print(f"\n[6/6] Assembling and saving final CSVs...")
print(f"Total models successfully processed for prediction: {models_processed_count}")
print(f"Total models skipped: {models_skipped_count}")

# Combine predicted deltas
if all_predicted_deltas:
    final_pred_df = pd.concat(all_predicted_deltas, ignore_index=True)
    # Drop duplicates that might arise if a patient/celltype/marker/stim combo was predicted in multiple folds (shouldn't happen with LOPO CV)
    final_pred_df = final_pred_df.drop_duplicates()
    print(f"Final predicted deltas DataFrame shape: {final_pred_df.shape}")
    # Ensure correct column order
    final_pred_df = final_pred_df[["patient", "cell_type", "marker", "stim", "median_diff"]]
    print(f"Saving predicted median differences to: {PRED_CSV_OUT}")
    final_pred_df.to_csv(PRED_CSV_OUT, index=False)
else:
    print("[WARN] No predicted delta rows were generated. Predicted CSV will be empty or not created.")

# Combine ground truth deltas
if all_ground_truth_deltas:
    final_gt_df = pd.concat(all_ground_truth_deltas, ignore_index=True)
    # Drop duplicates - GT calc is independent of folds, so same combo might appear N_FOLDS times
    final_gt_df = final_gt_df.drop_duplicates()
    print(f"Final ground truth deltas DataFrame shape: {final_gt_df.shape}")
    # Ensure correct column order
    final_gt_df = final_gt_df[["patient", "cell_type", "marker", "stim", "median_diff"]]
    print(f"Saving ground truth median differences to: {GT_CSV_OUT}")
    final_gt_df.to_csv(GT_CSV_OUT, index=False)
else:
    print("[WARN] No ground truth delta rows were generated. Ground truth CSV will be empty or not created.")

print("\nScript finished.")