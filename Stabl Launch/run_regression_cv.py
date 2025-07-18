#!/usr/bin/env python
"""
Script to run STABL with ElasticNet/Lasso/ALasso (and the corresponding grid-search
baseline) on onset of labor data using late fusion by stim.
It loads the features from:
  ./Data/ina_13OG_final_long_allstims_filtered.csv
and the outcome (DOS, regression target) from:
  ./Data/outcome_table_all_pre.csv

It splits the features into separate data frames based on the stim suffix
(i.e. ifna, il246, unstim, lps, gmcsf), then runs multi‐omicTraceback (most recent call last)
Results (including scores, plots and selected features) are saved to:
  ./Results
"""

import os
import sys
sys.path.insert(0, '../cellot/stablVMax')
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, RepeatedKFold, LeaveOneGroupOut
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.base import clone
from stabl.multi_omic_pipelines import multi_omic_stabl_cv,multi_omic_stabl_cv_noe
from stabl.stabl import Stabl
from stabl.adaptive import ALasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
# ---------------------------
# Data loading and preparation
# ---------------------------

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run STABL Regression CV with configurable inputs.")
parser.add_argument(
    "--features_path",
    required=True,
    help="Path to the input features CSV file (wide format)."
)

parser.add_argument(
    "--model_chosen",
    required=True,
    help="Model chosen for the regression (put None if classification)."
)
'''
parser.add_argument(
    "--stabl_chosen",
    required=True,
    help="Stabl model chosen for the regression (put None if classification)."
)
'''
parser.add_argument(
    "--results_dir",
    required=True,
    help="Base directory where results subdirectories will be created."
)
parser.add_argument(
    "--artificial_type",
    required=True,
    choices=["random_permutation", "knockoff"],
    help="Type of artificial features for STABL ('random_permutation' or 'knockoff')."
)
args = parser.parse_args()

# --- Use Parsed Arguments ---
features_path = args.features_path
artificial_type_arg = args.artificial_type
model_chosen=args.model_chosen

# Define outcome path (can remain hardcoded if it's always the same)
outcome_path = "./Data/outcome_table_all_pre.csv"

# Define output path dynamically based on inputs
input_stem = Path(features_path).stem 
results_path=args.results_dir
print(f"Input Features: {features_path}")
print(f"Results will be saved to: {results_path}")
print(f"Using STABL artificial type: {artificial_type_arg}")
os.makedirs(results_path, exist_ok=True)

# Load features and outcomes, index should be in column 0
df_features = pd.read_csv(features_path, index_col=0)
df_outcome = pd.read_csv(outcome_path, index_col=0, dtype={'DOS': int})
df_outcome = df_outcome[df_outcome.index.isin(df_features.index)]

y = df_outcome["DOS"]
df_features = df_features[df_features.index.isin(y.index)]
# ---------------------------
# Split features by stim
# ---------------------------
# We assume that all feature columns (except patient_id, now the index) end with one of:
# stims = ['ifna', 'il246', 'unstim', 'lps', 'gmcsf']
# stims = ['Unstim', 'TNFa', 'LPS', 'IL246', 'IFNa', 'GMCSF', 'PI', 'IL33']

if "unstim" in input_stem.lower(): # Check if 'unstim' is in the filename stem
    stims = ['Unstim']
    print("Detected Unstim-only input file. Using stims:", stims)
elif "medians_filtered" in input_stem.lower():
    stims = ['TNFa', 'LPS', 'IL246', 'IFNa', 'GMCSF', 'PI', 'IL33'] # Only stims, no Unstim
    print("Detected Filtered Predicted Stims input file. Using stims:", stims)
elif "ground_truth_features_filtered" in input_stem.lower(): 
    stims = ['Unstim', 'LPS', 'IL246', 'IFNa', 'GMCSF'] # Both Unstim and Stims, but not 'TNFa', 'PI', 'IL33'
    print("Detected Filtered Ground Truth (Unstim+Stims) input file. Using stims:", stims)
elif "combined_gt_pred" in input_stem.lower():
    stims = ['Unstim', 'LPS', 'IL246', 'IFNa', 'GMCSF','TNFapred', 'LPSpred', 'IL246pred', 'IFNapred', 'GMCSFpred', 'PIpred', 'IL33pred']
    print("Detected Filtered Ground Truth (Unstim+Stims) input file. Using stims:", stims)
else:
    print(f"Warning: Could not determine stims from filename stem '{input_stem}'. Using default full list.")
    stims = ['Unstim','TNFa', 'LPS', 'IL246', 'IFNa', 'GMCSF', 'PI', 'IL33']


# Create datadict with stim keys
data_dict = {}

for stim in stims:
    cols = [col for col in df_features.columns if col.endswith(stim)]
    if cols:
        data_dict[stim] = df_features[cols]
    else:
        print(f"Warning: No columns found for stim '{stim}'.")

if not data_dict:
    raise ValueError("No stim-specific features found. Please check your feature names.")

# ---------------------------
# Define cross-validation splits
# ---------------------------

groups = df_features.index.to_series().apply(lambda x: x.split('_')[0])
outer_cv = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

print(f"INFO: Using GroupShuffleSplit for outer CV ({groups.nunique()} groups/folds expected).") # Optional: Add print statement

# Inner CV for grid search: use RepeatedKFold
inner_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

# ---------------------------
# Define the estimators and grid search objects
# ---------------------------

lasso = Lasso(max_iter=int(1e6), random_state=42)
lasso_cv = GridSearchCV(
    lasso,
    param_grid={"alpha": np.logspace(-2, 2, 30)},
    scoring="r2",
    cv=inner_cv,
    n_jobs=-1
)

en = ElasticNet(max_iter=int(1e6), random_state=42)
en_cv = GridSearchCV(
    en,
    param_grid={"alpha": np.logspace(-2, 2, 10), "l1_ratio": [0.5, 0.7, 0.9]},
    scoring="r2",
    cv=inner_cv,
    n_jobs=-1
)

alasso = ALasso(max_iter=int(1e6), random_state=42)
alasso_cv = GridSearchCV(
    alasso,
    param_grid={"alpha": np.logspace(-2, 2, 30)},
    scoring="r2",
    cv=inner_cv,
    n_jobs=-1
)
rf = RandomForestRegressor(random_state=42, max_features=0.2)
xgb = XGBRegressor(random_state=42, importance_type="gain", objective="reg:squarederror")

stabl_lasso = Stabl(
    base_estimator=lasso,
    n_bootstraps=1000,
    artificial_type=artificial_type_arg,
    artificial_proportion=1.0, #artificial_proportion=1.0
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=0.5,
    random_state=42,
    lambda_grid={"alpha": np.logspace(0, 2, 10)}, #lambda_grid={"alpha": np.logspace(-3, 1, 15)}  np.logspace(0, 2, 10)
    verbose=1
)

stabl_alasso = clone(stabl_lasso).set_params(
    base_estimator=alasso,
    lambda_grid={"alpha": np.logspace(0, 2, 10)},
    verbose=1
)

stabl_en = clone(stabl_lasso).set_params(
    base_estimator=en,
    lambda_grid=[{"alpha": np.logspace(0.5, 2, 10), "l1_ratio": [0.5, 0.7, 0.9]}],
    verbose=1
)
xgb_grid = {"max_depth": [3, 6, 9], "reg_alpha": [0, 0.5, 1, 2]}
rf_grid = {"max_depth": [3, 5, 7, 9, 11]}
stabl_rf = clone(stabl_lasso).set_params(
    base_estimator=rf,
    lambda_grid=rf_grid,
    verbose=1
)

stabl_xgb = clone(stabl_lasso).set_params(
    base_estimator=xgb,
    lambda_grid=[xgb_grid],
    verbose=1
)

estimators = {
    "lasso": lasso_cv,
    "alasso": alasso_cv,
    "en": en_cv,
    "stabl_lasso": stabl_lasso,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en,
}

models = [
    "STABL Lasso",
    "STABL ALasso",
    "STABL ElasticNet"
]


print("Starting multi-omic STABL CV with late fusion...")
predictions_dict = multi_omic_stabl_cv(
    data_dict=data_dict,
    y=y,
    outer_splitter=outer_cv,
    estimators=estimators,
    task_type="regression",
    model_chosen=model_chosen,
    save_path=results_path,
    models=models,
    outer_groups=groups,       
    early_fusion=False, 
    late_fusion=False,
    n_iter_lf=100000
)

print("STABL CV finished.")
print("Results (including r2, r, rmse, selected features, and plots) have been saved to:")
print(results_path)