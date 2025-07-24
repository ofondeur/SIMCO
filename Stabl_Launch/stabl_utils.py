# stabl_utils.py

import os
import sys
sys.path.insert(0, '../Stabl')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.model_selection import GroupShuffleSplit, RepeatedKFold, GridSearchCV
from stabl.stabl import Stabl
from stabl.adaptive import ALasso
from sklearn.base import clone

def get_estimators(artificial_type):
    lasso = Lasso(max_iter=int(1e6), random_state=42)
    en = ElasticNet(max_iter=int(1e6), random_state=42)
    alasso = ALasso(max_iter=int(1e6), random_state=42)

    inner_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

    lasso_cv = GridSearchCV(lasso, param_grid={"alpha": np.logspace(-2, 2, 30)}, scoring="r2", cv=inner_cv, n_jobs=-1)
    en_cv = GridSearchCV(en, param_grid={"alpha": np.logspace(-2, 2, 10), "l1_ratio": [0.5, 0.7, 0.9]}, scoring="r2", cv=inner_cv, n_jobs=-1)
    alasso_cv = GridSearchCV(alasso, param_grid={"alpha": np.logspace(-2, 2, 30)}, scoring="r2", cv=inner_cv, n_jobs=-1)

    stabl_lasso = Stabl(
        base_estimator=lasso,
        n_bootstraps=1000,
        artificial_type=artificial_type,
        artificial_proportion=1.0,
        replace=False,
        fdr_threshold_range=np.arange(0.1, 1, 0.01),
        sample_fraction=0.5,
        random_state=42,
        lambda_grid={"alpha": np.logspace(0, 2, 10)},
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

    estimators = {
        "lasso": lasso_cv,
        "alasso": alasso_cv,
        "en": en_cv,
        "stabl_lasso": stabl_lasso,
        "stabl_alasso": stabl_alasso,
        "stabl_en": stabl_en,
    }

    return estimators

def split_features_by_stim(df_features, stims):
    data_dict = {}
    for stim in stims:
        cols = [col for col in df_features.columns if col.endswith(stim)]
        if cols:
            data_dict[stim] = df_features[cols]
        else:
            print(f"Warning: No columns found for stim '{stim}'.")
    return data_dict

def get_stims(input_stem):
    if "unstim_only" in input_stem.lower():
        print("Detected Unstim-only input file. Using stims:", stims)
        return ['Unstim']
    elif "no_unstim" in input_stem.lower():
        print("Detected only Stims input file. Using stims:", stims)
        return ['TNFa', 'LPS', 'IL246', 'IFNa', 'GMCSF', 'PI', 'IL33']
    elif "merged" in input_stem.lower():
        print("Detected merged (OOL+ predicted stims) input file. Using stims:", stims)
        stims = ['Unstim','TNFa', 'LPS', 'IL246', 'IFNa', 'GMCSF', 'PI', 'IL33']
        return [f"{stim}_OOL" for stim in stims] + [f"{stim}_CellOT" for stim in stims]
    else:
        print(f"Warning: Could not determine stims from filename stem '{input_stem}'. Using default full list.")
        return ['Unstim','TNFa', 'LPS', 'IL246', 'IFNa', 'GMCSF', 'PI', 'IL33']

def process_data(features_path, outcome_path):
    df_features = pd.read_csv(features_path, index_col=0)
    df_outcome = pd.read_csv(outcome_path, index_col=0, dtype={'DOS': int})
    df_outcome = df_outcome[df_outcome.index.isin(df_features.index)]

    y = df_outcome["DOS"]
    df_features = df_features[df_features.index.isin(y.index)]
    return df_features, y

