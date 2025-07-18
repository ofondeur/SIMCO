#!/usr/bin/env python
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '../cellot/stablVMax')
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.metrics import mean_absolute_error,mean_squared_error
import shutil
import itertools
# Import scikit-learn modules
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, RepeatedKFold, LeaveOneGroupOut
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.base import clone
from xgboost import XGBRegressor
from stabl.cross_validation_drug_vs_dmso import cv_on_existing_feats
from stabl.stabl import Stabl
from sklearn.ensemble import RandomForestRegressor
from stabl.adaptive import ALasso

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Run STABL Regression CV with configurable inputs.")
parser.add_argument(
    "--features_path",
    required=True,
    help="Path to the input features CSV file (wide format)."
)
parser.add_argument(
    "--fold_feats_path",
    required=True,
    help="Path to the features selected per fold."
)
parser.add_argument(
    "--model_chosen",
    required=True,
    help="Model chosen for the regression (put None if classification)."
)

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

# Define outcome path
outcome_path = "/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/outcome_table_all_pre.csv"

# Define output path dynamically based on inputs
input_stem = Path(features_path).stem # Get filename without extension

results_path=args.results_dir
fold_feats_path=args.fold_feats_path
print(f"Input Features: {features_path}")
print(f"Results will be saved to: {results_path}")
print(f"Using STABL artificial type: {artificial_type_arg}")
os.makedirs(results_path, exist_ok=True)


# Load features and outcomes
df_features = pd.read_csv(features_path, index_col=0)
df_outcome = pd.read_csv(outcome_path, index_col=0, dtype={'DOS': int})
df_outcome = df_outcome[df_outcome.index.isin(df_features.index)]

# Outcome is in the column 'DOS'
y = df_outcome["DOS"]

# Ensure the features dataframe contains only patients with available outcomes
df_features = df_features[df_features.index.isin(y.index)]

# ---------------------------
# Split features by stim
# ---------------------------

stims = ['Unstim','TNFa', 'LPS', 'IL246', 'IFNa', 'GMCSF', 'PI', 'IL33']
stims_with_suffixes = [f"{stim}_OOL" for stim in stims] + [f"{stim}_CellOT" for stim in stims]

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

print(f"INFO: Using GroupShuffleSplit for outer CV ({groups.nunique()} groups/folds expected).")

# Inner CV for grid search: use RepeatedKFold
inner_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

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

stabl_lasso = Stabl(
    base_estimator=lasso,
    n_bootstraps=1000,
    artificial_type=artificial_type_arg,
    artificial_proportion=1.0,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=0.5,
    random_state=42,
    lambda_grid={"alpha": np.logspace(0, 2, 10)},
    verbose=1
)

# Create STABL models for Lasso, ALasso and ElasticNet
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


estimators = {
    "lasso": lasso_cv,
    "alasso": alasso_cv,
    "en": en_cv,
    "stabl_lasso": stabl_lasso,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en,
}

models = [
    "STABL ALasso",
    "STABL Lasso",
    "STABL ElasticNet",
]

# Define parameters grid for XGBoost
param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [2, 4, 10],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.5, 0.7],
    "colsample_bytree": [0.5,0.8],
    'gamma': [0,1], 
    'reg_alpha': [0], 
    'reg_lambda': [1]
}

all_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

best_scores = {model: float("inf") for model in models}
best_combinations = {model: None for model in models}
best_paths = {model: None for model in models}

for i, param_values in enumerate(all_combinations):
    param_dict = dict(zip(param_names, param_values))
    print(f"\n==== Grid Search Iteration {i+1}/{len(all_combinations)} ====")
    print("Trying parameters:", param_dict)

    xgboost_model = XGBRegressor(**param_dict, verbosity=0)
    estimators["xgboost"] = xgboost_model
    
    tmp_path = Path(results_path) / f"gridsearch_tmp_{i}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Launch prediction for this set of parameter
    predictions_dict = cv_on_existing_feats(
        data_dict=data_dict,
        y=y,
        outer_splitter=outer_cv,
        estimators=estimators,
        task_type="regression",
        model_chosen=model_chosen,
        fold_feats_path=fold_feats_path,
        save_path=tmp_path,
        models=models,
        outer_groups=groups,       
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=100000
        )
    keep_tmp = False
    for model in models:
        true_y = y.loc[predictions_dict[model].index]
        pred_y = predictions_dict[model]
        mae = mean_absolute_error(true_y, pred_y)
        print(f"{model} → MAE: {mae:.4f}")
        rmse = np.sqrt(mean_squared_error(true_y, pred_y))
        print(f"{model} → RMSE: {rmse:.4f}")
    
        if rmse < best_scores[model]:
            best_scores[model] = rmse
            best_combinations[model] = param_dict
            best_paths[model] = tmp_path
            keep_tmp = True
    
    if not keep_tmp:
        shutil.rmtree(tmp_path)

print("\n===== Best RMSEs per model =====")
for model in models:
    print(f"{model} → RMSE: {best_scores[model]:.4f}, Best Params: {best_combinations[model]}")

for model in models:
    final_model_path = Path(results_path) / model.replace(" ", "_")
    if final_model_path.exists():
        shutil.rmtree(final_model_path)
    shutil.copytree(best_paths[model], final_model_path)
    print(f"→ Best results for {model} saved to {final_model_path}")
