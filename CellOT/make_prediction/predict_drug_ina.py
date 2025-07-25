import os
import anndata as ad
from prediction_utils import load_data_networks_drug_OOL
from cellot.data.cell import AnnDataDataset
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import gc

drug_used = 'MAP'
model = 'original_13_HVPV'

def predict_per_patient_drug_response(result_path, unstim_data_path, stim, model, cell_type, patient):
    anndata_to_predict, g, features_eval, features_input, semisuffled_features= load_data_networks_drug_OOL(
        result_path=result_path,
        unstim_data_path=unstim_data_path,
        stim=stim,
        cell_type=cell_type
    )
    
    untreated_anndata_to_predict = anndata_to_predict[:, features_input].copy() # filter the input on the markers we want to use to predict
    stims_in_data=untreated_anndata_to_predict.obs['stim'].unique().tolist()
    drugs_in_data=untreated_anndata_to_predict.obs['drug'].unique().tolist()
    cell_type_in_data=untreated_anndata_to_predict.obs['cell_type'].unique().tolist()
    true_medians = np.median(untreated_anndata_to_predict.X, axis=0)
    true_result = {
        f"{cell_type}_{marker}_Unstim": val
        for marker, val in zip(features_eval, true_medians)
    }
    print(f"The categories of the anndata before the source, target split is: stim {stims_in_data}, drugs {drugs_in_data}, cells {cell_type_in_data}, with a shape of {untreated_anndata_to_predict.X.shape}")
    dataset_args = {}
    dataset = AnnDataDataset(untreated_anndata_to_predict.copy(), **dataset_args) #transform the dataset to the expected format
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))
    outputs = g.transport(inputs.requires_grad_(True)).detach().numpy()
    
    predicted = ad.AnnData(
        outputs,
        obs=dataset.adata.obs.copy(),
        var=dataset.adata.var.copy(),
    )
    predicted = predicted[:, semisuffled_features]
    predicted.var_names = features_eval
    predicted.obs["stim"] = stim
    predicted.obs["state"] = "predicted"
    predicted.obs["sampleID"] = patient
    predicted.obs["cell_type"] = cell_type
    predicted.obs["drug"] = drug_used

    pred_medians = np.median(predicted.X, axis=0)
    pred_result = {
        f"{cell_type}_{marker}_{stim}": val
        for marker, val in zip(features_eval, pred_medians)
    }

    result = {"Individual": patient}
    result.update(pred_result)
    result.update(true_result)
    return result

# Load fold info
FOLD_INFO_FILE = f"../datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info_original_median_20marks.csv"
fold_info = defaultdict(lambda: defaultdict(dict))
with open(FOLD_INFO_FILE, 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        stim, sanitized_ct, original_ct, fold_idx, *patients = parts
        fold_info[stim][sanitized_ct][int(fold_idx)] = patients

MODEL_BASE_DIR = f"../results/cross_validation_{model}"

print(f"[INFO] Starting predictions")
all_results = []
csv_path = f"../results/ina_13OG_{drug_used}_2.csv"
path_patients_ina='../datasets/ptb_concatenated_per_condition_celltype/patients_ina.txt'
with open(path_patients_ina, 'r') as f:
    patients_ina_list = [line.strip() for line in f if line.strip()]
existing_patients = set()
first_write = not os.path.exists(csv_path)

if not first_write:
    try:
        df_existing = pd.read_csv(csv_path)
        existing_patients = set(df_existing["Individual"].astype(str))
        print(f"[INFO] {len(existing_patients)} patients already present in {csv_path}")
    except Exception as e:
        print(f"[WARN] Failed to read existing CSV: {e}")
        first_write = True

OUTPUT_DIR = "../datasets/ool_by_patient/unstim_final"
with open(csv_path, "a") as f:
    for patient in patients_ina_list:
        if patient in existing_patients:
            print(f"[SKIP] Already done patient {patient}")
            continue

        patient_row = {"Individual": patient}
        for stim in fold_info:
            for cell_type in fold_info[stim]:
                result_path = f"../results_drug/cross_validation_drug_OG13/{drug_used}/{stim}/{cell_type}/model-{stim}_{cell_type}"
                unstim_data_path = f"{MODEL_BASE_DIR}/{stim}/{cell_type}/model-{stim}_{cell_type}/preds_ina/{patient}_{cell_type}_{stim}.h5ad"
                model_file = os.path.join(result_path, "cache/model.pt")
                if not os.path.exists(model_file):
                    print(f"[SKIP] No model for {stim} / {cell_type}")
                    continue
                if not os.path.exists(unstim_data_path):
                    print(f"[SKIP] No data for {stim} / {cell_type}, {unstim_data_path}")
                    continue
                row = predict_per_patient_drug_response(result_path, unstim_data_path, stim, model, cell_type, patient)
                if row:
                    del row["Individual"]
                    patient_row.update(row)
                    print(f"[INFO] Done for {stim}, {cell_type}, {patient}")
                else:
                    print(f"[WARN] No row for {stim}, {cell_type}, {patient}")
                gc.collect()

        if len(patient_row) > 1:
            if first_write:
                f.write(','.join(patient_row.keys()) + '\n')
                first_write = False
            f.write(','.join(str(patient_row[k]) for k in patient_row.keys()) + '\n')
            existing_patients.add(patient)
            print(f"[WRITE] Patient {patient} written.")
        else:
            print(f"[SKIP] No prediction written for patient {patient}")
        gc.collect()