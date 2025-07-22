# stabl_utils.py

import os
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
