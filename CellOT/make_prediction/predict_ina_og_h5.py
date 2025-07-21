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

drug_used = 'DMSO'
model = 'original_13_HVPV'

celltype_name_map = {
    'Bcells': 'Bcells',
    'CD4Tcm': 'CD4Tcm',
    'CD4Tcm_CCR2pos': 'CD4Tcm CCR2+',
    'CD4Teff': 'CD4Teff',
    'CD4Tem': 'CD4Tem',
    'CD4Tem_CCR2pos': 'CD4Tem CCR2+',
    'CD4Tnaive': 'CD4Tnaive',
    'CD4Tregs': 'CD4Tregs',
    'CD4negCD8negTcells': 'CD4negCD8negTcells',
    'CD56hiCD16negNK': 'CD56hiCD16negNK',
    'CD56loCD16posNK': 'CD56loCD16posNK',
    'CD8Tcells_Th1': 'CD8Tcells Th1',
    'CD8Tcm': 'CD8Tcm',
    'CD8Tcm_CCR2pos': 'CD8Tcm CCR2+',
    'CD8Teff': 'CD8Teff',
    'CD8Tem': 'CD8Tem',
    'CD8Tem_CCR2pos': 'CD8Tem CCR2+',
    'CD8Tnaive': 'CD8Tnaive',
    'Granulocytes': 'Granulocytes',
    'MDSCs': 'MDSCs',
    'NK_cells_CD11cpos': 'NK cells CD11c+',
    'NK_cells_CD11cneg': 'NK cells CD11c-',
    'NKT': 'NKT',
    'cMCs': 'cMCs',
    'intMCs': 'intMCs',
    'mDCs': 'mDCs',
    'ncMCs': 'ncMCs',
    'pDCs': 'pDCs'
}

def load_patient_anndata(data_path, patient, cell_type, drug_prefix):
    adata = ad.read_h5ad(data_path, backed='r')  # accès sur disque
    mask = (
        (adata.obs['stim'] == 'Unstim') &
        (adata.obs['cell_type'] == celltype_name_map[cell_type]) &
        (adata.obs['drug'].str.startswith(drug_prefix)) &
        (adata.obs['sampleID'] == patient)
    )
    
    subset = adata[mask].to_memory()
    adata.file.close()
    return subset
    
def predict_per_patient(result_path, unstim_data_path, stim, model, cell_type, patient, output_dir):
    config_path = os.path.join(result_path, "config.yaml")
    chkpt = os.path.join(result_path, "cache/model.pt")

    feats_input_path = os.path.join(result_path, "features_input_names.txt")
    feats_eval_path = os.path.join(result_path, "features_eval_names.txt")
    semisuffled_features_path = os.path.join(result_path, "semisuffled_features.txt")
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
    print('Model loaded, beginning prediction', flush=True)
    anndata = load_patient_anndata(
        data_path=unstim_data_path,
        patient=patient,
        cell_type=cell_type,
        drug_prefix=drug_used,
    )

    if anndata.shape[0] == 0:
        print(f"[WARN] No data for patient {patient}, stim {stim}, cell {cell_type}", flush=True)
        return
    anndata = anndata[:, features_input]
    dataset_args = {}
    dataset = AnnDataDataset(anndata.copy(), **dataset_args)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))
    outputs = g.transport(inputs.requires_grad_(True)).detach().numpy()

    predicted = ad.AnnData(outputs, obs=dataset.adata.obs.copy())
    predicted = predicted[:, :len(semisuffled_features)]
    predicted.var_names = features_eval
    predicted.obs["stim"] = stim
    predicted.obs["state"] = "predicted"
    predicted.obs["sampleID"] = patient
    predicted.obs["cell_type"] = cell_type

    # Save the prediction
    output_filename = f"{patient}_{cell_type}_{stim}.h5ad"
    output_path = os.path.join(output_dir, output_filename)
    predicted.write(output_path)
    print(f"[WRITE] Saved: {output_path}", flush=True)

print(f"[INFO] Loading fold info")
# Load fold info
FOLD_INFO_FILE = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info_original_median_20marks.csv"
fold_info = defaultdict(lambda: defaultdict(dict))
with open(FOLD_INFO_FILE, 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        stim, sanitized_ct, original_ct, fold_idx, *patients = parts
        fold_info[stim][sanitized_ct][int(fold_idx)] = patients


MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}"

print(f"[INFO] Starting predictions")

path_patients_ina='/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/patients_ina.txt'
with open(path_patients_ina, 'r') as f:
    patients_ina_list = [line.strip() for line in f if line.strip()]
    
print(f"[INFO] Starting to predict", flush=True)
OUTPUT_DIR_2 = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ool_by_patient/unstim_final"
for patient in patients_ina_list:
    for stim in fold_info:
        for cell_type in fold_info[stim]:
            unstim_data_path =  os.path.join(OUTPUT_DIR_2, f"{patient}_Unstim.h5ad")
            output_dir=f"{MODEL_BASE_DIR}/{stim}/{cell_type}/model-{stim}_{cell_type}/preds_ina"
            os.makedirs(output_dir, exist_ok=True)
            result_path = f"{MODEL_BASE_DIR}/{stim}/{cell_type}/model-{stim}_{cell_type}"
            model_file = os.path.join(result_path, "cache/model.pt")
            if not os.path.exists(model_file):
                print(f"[SKIP] No model for {stim} / {cell_type}", flush=True)
                continue
            output_filename = f"{patient}_{cell_type}_{stim}.h5ad"
            output_path = os.path.join(output_dir, output_filename)
            try:
                if os.path.exists(output_path):
                    print(f'Already done {output_filename}')
                else:
                    predict_per_patient(result_path, unstim_data_path, stim, model, cell_type, patient, output_dir)
            except Exception as e:
                print(f"[ERROR] Exception for {patient}-{stim}-{cell_type}: {e}", flush=True)

    gc.collect()

print(f"[DONE] All predictions saved to {output_dir}", flush=True)