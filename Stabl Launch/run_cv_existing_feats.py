#!/usr/bin/env python
import os
import sys
sys.path.insert(0, '/home/groups/gbrice/ptb-drugscreen/ot/cellot/stablVMax')
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json
from xgboost import XGBRegressor

from sklearn.model_selection import GroupShuffleSplit

from stabl.cross_validation_drug_vs_dmso import cv_drug_vs_dmso
from stabl_utils import get_estimators,split_features_by_stim
# ---------------------------
# Data loading and preparation
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Run STABL Regression CV with configurable inputs.")
    parser.add_argument(
        "--notreat_features_path",
        required=True,
        help="Path to the not treated features CSV file (wide format)."
    )

    parser.add_argument(
        "--fold_feats_path",
        required=True,
        help="Path to the features selected per fold."
    )
    parser.add_argument('--xgb_config_path', type=str, default=None)
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
    notreat_features_path = args.notreat_features_path
    drug_to_use_list = ['PRA','LPZ','SALPZ','SA','MF','CHT','THF','RIF','MAP']
    artificial_type_arg = args.artificial_type # Store the argument value
    model_chosen=args.model_chosen
    outcome_path = "/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/outcome_table_all_pre.csv"

    input_stem = Path(notreat_features_path).stem
    results_path=args.results_dir
    fold_feats_path=args.fold_feats_path
    print(f"Results will be saved to: {results_path}")
    print(f"Using STABL artificial type: {artificial_type_arg}")
    os.makedirs(results_path, exist_ok=True)

    df_features_no_treat = pd.read_csv(notreat_features_path, index_col=0)
    df_outcome = pd.read_csv(outcome_path, index_col=0, dtype={'DOS': int})
    df_outcome = df_outcome[df_outcome.index.isin(df_features_no_treat.index)]
    # Outcome is in the column 'DOS'
    y = df_outcome["DOS"]

    df_features_no_treat = df_features_no_treat[df_features_no_treat.index.isin(y.index)]
    # ---------------------------
    # Split features by stim
    # ---------------------------

    stims = ['Unstim','TNFa', 'LPS', 'IL246', 'IFNa', 'GMCSF', 'PI', 'IL33'] #'Unstim', 


    no_treat_dict = {}
    for stim in stims:
        cols = [col for col in df_features_no_treat.columns if col.endswith(stim)]
        if cols:
            no_treat_dict[stim] = df_features_no_treat[cols]
        else:
            print(f"Warning: No columns found for stim '{stim}'.")
        
    if not no_treat_dict:
        raise ValueError("No stim-specific features found. Please check your feature names.")
        
    data_dict={}
    data_dict['No_treat']=no_treat_dict
    for drug in drug_to_use_list:
        if drug not in ['SA', 'RIF', 'SALPZ', 'CHT', 'THF', 'LPZ', 'MAP', 'PRA', 'MF']:
            print(f"[WARNING] Skipping drug {drug}, name is not usual")
            continue
        path_features_drug=f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/ina_13OG_{drug}_untreated_unstim.csv"
        if not os.path.exists(path_features_drug):
            print(f"[WARNING] Skipping drug {drug}, path {path_features_drug} not found")
            continue
        df_features_treat = pd.read_csv(path_features_drug, index_col=0)
        df_features_treat = df_features_treat[df_features_treat.index.isin(y.index)]
        print(df_features_treat.shape)
        treat_dict=split_features_by_stim(df_features_treat, stims)
        if not treat_dict:
            raise ValueError("No stim-specific features found. Please check your feature names.")
        data_dict[drug]=treat_dict


    groups = df_features_no_treat.index.to_series().apply(lambda x: x.split('_')[0])

    outer_cv = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

    #outer_cv = LeaveOneGroupOut()
    print(f"INFO: Using LeaveOneGroupOut for outer CV ({groups.nunique()} groups/folds expected).") # Optional: Add print statement
    estimators = get_estimators(artificial_type_arg)
    models = [
        "STABL Lasso",
        "STABL ALasso",
        "STABL ElasticNet"
    ]
    if args.model_chosen == "xgboost" and args.xgb_config_path is not None:
        with open(args.xgb_config_path, 'r') as f:
            xgb_params = json.load(f)
        xgboost_model = XGBRegressor(**xgb_params, verbosity=0)
        estimators["xgboost"] = xgboost_model
        
    print("Starting multi-omic STABL CV with late fusion...")
    results_path = os.path.join(args.results_dir, f"results_no_treatment") # Construct unique path

    predictions_dict = cv_drug_vs_dmso(
        data_dict=data_dict,
        y=y,
        outer_splitter=outer_cv,
        estimators=estimators,
        task_type="regression",
        model_chosen=model_chosen,
        fold_feats_path=fold_feats_path,
        save_path=results_path,
        models=models,
        outer_groups=groups,       
        early_fusion=False,
        late_fusion=False,
        n_iter_lf=100000
    )

    prediction_csv_path=os.path.join(results_path, "Training CV/No_treat/STABL ALasso/STABL ALasso predictions.csv")
    prediction_final=pd.read_csv(prediction_csv_path)
    for drug in drug_to_use_list:
        prediction_csv_path=os.path.join(results_path, f"Training CV/{drug}/STABL ALasso/STABL ALasso predictions.csv")
        prediction_df=pd.read_csv(prediction_csv_path)
        prediction_final[drug]=prediction_df.iloc[:,1]
    prediction_csv_path=os.path.join(results_path, 'prediction_drugs_final.csv')
    prediction_final.to_csv(prediction_csv_path)
    print("CV for no treatement finished.")
    print("Predictions for each patient and all the drugs have been saved to:")
    print(prediction_csv_path)

if __name__ == "__main__":
    main()