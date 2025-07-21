import os
import anndata as ad
from cellot.utils.helpers import load_config
from cellot.utils.loaders import load
from cellot.models.cellot import load_networks
from cellot.data.cell import read_list
from cellot.data.cell import AnnDataDataset
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import gc
from scipy import sparse

model = 'original_13_HVPV'

def predict_per_patient(result_path, unstim_data_path, stim, model, cell_type, patient,drug_used):
    config_path = os.path.join(result_path, "config.yaml")
    chkpt = os.path.join(result_path, "cache/model.pt")
    
    feats_input_path= os.path.join(result_path, "features_input_names.txt")
    feats_eval_path= os.path.join(result_path, "features_eval_names.txt")
    semisuffled_features_path= os.path.join(result_path, "semisuffled_features.txt")
    
    if os.path.exists(feats_eval_path):
        features_eval = read_list(feats_eval_path)
        features_input = read_list(feats_input_path)
        semisuffled_features = read_list(semisuffled_features_path)
    else:
        features_eval = read_list('/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/13features.txt')
        features_input = features_eval
        semisuffled_features = features_eval
    config = load_config(config_path)
    if not Path(chkpt).exists():
        print(f"[ERROR] Checkpoint missing at: {chkpt}", flush=True)
        return
    model_kwargs = {}
    model_kwargs["input_dim"] = len(features_input)
    restore=chkpt
    _, g = load_networks(config, **model_kwargs)
    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        g.load_state_dict(ckpt["g_state"])
    g.eval()
    
    # load the data to predict
    anndata_to_predict = ad.read_h5ad(unstim_data_path)
    
    anndata_to_predict=anndata_to_predict[(anndata_to_predict.obs['stim']==stim)&(anndata_to_predict.obs['cell_type']==cell_type)].copy()
    
    untreated_anndata_to_predict = anndata_to_predict[:, features_input].copy() # filter the input on the markers we want to use to predict
    stims_in_data=untreated_anndata_to_predict.obs['stim'].unique().tolist()
    drugs_in_data=untreated_anndata_to_predict.obs['drug'].unique().tolist()
    cell_type_in_data=untreated_anndata_to_predict.obs['cell_type'].unique().tolist()
    
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
    return result
cells_list=['Bcells', 'CD4Tcm', 'CD4Tcm CCR2+', 'CD4Teff', 'CD4Tem', 'CD4Tem CCR2+', 'CD4Tnaive', 'CD4Tregs', 'CD4negCD8negTcells', 'CD56hiCD16negNK', 'CD56loCD16posNK', 'CD8Tcells Th1', 'CD8Tcm', 'CD8Tcm CCR2+', 'CD8Teff', 'CD8Tem', 'CD8Tem CCR2+', 'CD8Tnaive', 'Granulocytes', 'MDSCs', 'NK cells CD11c+', 'NK cells CD11c-', 'NKT', 'cMCs', 'intMCs', 'mDCs', 'ncMCs', 'pDCs']
MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_unstim/cross_validation_{model}"

print(f"[INFO] Starting predictions")
def sanitize_name(name):
    """Sanitizes names for use in paths/filenames."""
    name = str(name)
    name = name.replace(' ', '_')
    name = name.replace('/', '-')
    name = name.replace('+', 'pos')
    name = name.replace('-', 'neg')
    name = name.replace('NK_cells_', 'NK_cells_')
    return name
for drug_used in ['SA', 'RIF', 'SALPZ', 'CHT', 'THF', 'LPZ', 'MAP', 'PRA', 'MF']:
    all_results = []
    csv_path = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/ina_13OG_{drug_used}_unstim.csv"
    path_patients_ina='/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/patients_ina.txt'
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
            
    OUTPUT_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ool_by_patient/unstim_final"
    stim='Unstim'
    with open(csv_path, "a") as f:
        for patient in patients_ina_list:
            if patient in existing_patients:
                print(f"[SKIP] Already done patient {patient}")
                continue
    
            patient_row = {"Individual": patient}
            for cell_type in cells_list:
                sanitized_cell_type=sanitize_name(cell_type)
                result_path = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_unstim/cross_validation_drug_OG13/{drug_used}/{sanitized_cell_type}/model-Unstim_{sanitized_cell_type}"
                unstim_data_path = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ool_by_patient/unstim_final/{patient}_Unstim.h5ad"
                model_file = os.path.join(result_path, "cache/model.pt")
                if not os.path.exists(model_file):
                    print(f"[SKIP] No model for {stim} / {cell_type}, {model_file}",flush=True)
                    continue
                if not os.path.exists(unstim_data_path):
                    print(f"[SKIP] No data for {stim} / {cell_type}, {unstim_data_path}",flush=True)
                    continue
                row = predict_per_patient(result_path, unstim_data_path, stim, model, cell_type, patient,drug_used)
                if row:
                    del row["Individual"]
                    patient_row.update(row)
                    print(f"[INFO] Done for {stim}, {cell_type}, {patient}",flush=True)
                else:
                    print(f"[WARN] No row for {stim}, {cell_type}, {patient}",flush=True)
                gc.collect()
    
            if len(patient_row) > 1:  # on a prédit au moins une cellule pour ce patient
                if first_write:
                    f.write(','.join(patient_row.keys()) + '\n')
                    first_write = False
                f.write(','.join(str(patient_row[k]) for k in patient_row.keys()) + '\n')
                existing_patients.add(patient)
                print(f"[WRITE] Patient {patient} written.",flush=True)
            else:
                print(f"[SKIP] No prediction written for patient {patient}",flush=True)
            gc.collect()
    print(f"[WRITE] Drug {drug_used} finished.",flush=True)