import gc
from cellot.data.cell import read_list
from sklearn.metrics.pairwise import rbf_kernel
import os
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cellot.utils.helpers import load_config
from cellot.utils.loaders import load
from cellot.models.cellot import load_networks
from cellot.data.cell import read_list
from cellot.data.cell import AnnDataDataset
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import sys
from sklearn.metrics import r2_score
from collections import defaultdict

def load_patient_anndata(data_path, patient, cell_type, drug_prefix):
    adata = ad.read_h5ad(data_path, backed='r')  # accès sur disque
    mask = (
        (adata.obs['stim'] == 'Unstim') &
        (adata.obs['drug'].str.startswith(drug_prefix)) &
        (adata.obs['patient'] == patient)
    )
    
    subset = adata[mask].to_memory()
    adata.file.close()
    return subset
    
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
    
    
def predict_per_patient(result_path, unstim_data_path, stim, model, cell_type, patient):
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
    anndata = load_patient_anndata(
        data_path=unstim_data_path,
        patient=patient,
        cell_type=cell_type,
        drug_prefix=drug_used,
    )

    if anndata.shape[0] == 0:
        print(f"[WARN] No data for patient {patient}, stim {stim}, cell {cell_type}", flush=True)
        return

    anndata = anndata[:, features_input].copy()
    dataset_args = {}
    dataset = AnnDataDataset(anndata.copy(), **dataset_args) #transform the dataset to the expected format
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
    return predicted


drug_used = 'DMSO'
model = 'original1'
MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}"
FOLD_INFO_FILE = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info_original1.csv"
DATASET_BASE = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype"

markers = ['CD25', 'HLADR','IkB','pCREB','pERK','pMK2','pNFkB','pS6','pSTAT1','pSTAT3','pSTAT5','pSTAT6','pp38']

def compute_mae(true, pred):
    mae = np.abs(np.median(true) - np.median(pred))
    return mae

fold_info = defaultdict(lambda: defaultdict(dict))
with open(FOLD_INFO_FILE, 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        stim, sanitized_ct, original_ct, fold_idx, *patients = parts
        fold_info[stim][sanitized_ct][int(fold_idx)] = patients

results = []
median_corrs = []
for stim in ['IL246']:
    for cell_type in fold_info[stim]:
        print(f"Doing {cell_type} and {stim}")
        for marker in markers:
            medians_true = []
            medians_cellot = []
            medians_identity = []
            medians_returntrain = []
            for fold_idx in [0,1,2,3]:
                test_patients = fold_info[stim][cell_type][fold_idx]
                train_patients = [p for i, plist in fold_info[stim][cell_type].items() if i != fold_idx for p in plist]
                data_path = f"{DATASET_BASE}/{cell_type}_HV.h5ad"
                data = ad.read_h5ad(data_path)
                train_data = data[(data.obs['patient'].isin(train_patients)) & (data.obs['stim'] == stim)]
                result_path = f"{MODEL_BASE_DIR}/{stim}/{cell_type}/model-{stim}_{cell_type}_fold{fold_idx}"
                for patient in test_patients:
                    print(f"[INFO] Processing {stim}, {cell_type}, fold {fold_idx}, marker {marker}")
                    patient_data = data[data.obs['patient'] == patient]
                    try:
                        true_stim = patient_data[patient_data.obs['stim'] == stim][:, marker].X.flatten()
                        unstim = patient_data[patient_data.obs['stim'] == "Unstim"][:, marker].X.flatten()
        
                        pred = predict_per_patient(result_path, data_path, stim, model, cell_type, patient)
                        true_stim_vals = data[(data.obs['patient'] == patient) & (data.obs['stim'] == stim)][:, marker].X.flatten()
                        
                        pred_vals = pred[:, marker].X.flatten()
                        unstim_vals = data[(data.obs['patient'] == patient) & (data.obs['stim'] == "Unstim")][:, marker].X.flatten()
                        train_vals = train_data[:, marker].X.flatten()
                        train_median_val = np.median(train_vals)
                        
                        medians_true.append(np.median(true_stim_vals))
                        medians_cellot.append(np.median(pred_vals))
                        medians_identity.append(np.median(unstim_vals))
                        medians_returntrain.append(train_median_val)
                    
                    except Exception as e:
                        print(f"[WARNING] Error for patient {patient}, marker {marker}: {e}")
            if len(medians_true) == 8:
                r_cellot = np.corrcoef(medians_true, medians_cellot)[0, 1]
                r_identity = np.corrcoef(medians_true, medians_identity)[0, 1]
                r_returntrain = np.corrcoef(medians_true, medians_returntrain)[0, 1]
                r2_cellot = r2_score(medians_true, medians_cellot)
                r2_identity = r2_score(medians_true, medians_identity)
                r2_returntrain = r2_score(medians_true, medians_returntrain)
                median_corrs.append({
                    "stim": stim,
                    "cell_type": cell_type,
                    "marker": marker,
                    "model": "CellOT",
                    "corr": r_cellot,
                    "r2": r2_cellot
                })
                median_corrs.append({
                    "stim": stim,
                    "cell_type": cell_type,
                    "marker": marker,
                    "model": "Identity",
                    "corr": r_identity,
                    "r2": r2_identity
                })
                median_corrs.append({
                    "stim": stim,
                    "cell_type": cell_type,
                    "marker": marker,
                    "model": "ReturnTrain",
                    "corr": r_returntrain,
                    "r2": r2_returntrain
                })
            else:
                print(len(medians_true))
                continue
os.makedirs(f"{MODEL_BASE_DIR}/plots_corr_8dots", exist_ok=True)
outdir_path=f"{MODEL_BASE_DIR}/plots_corr_8dots/correlations.csv"

df_corrs = pd.DataFrame(median_corrs)
df_corrs.to_csv(outdir_path)
sns.boxplot(data=df_corrs, x="model", y="corr", palette="Set2")
plt.title("Correlation of median marker values per patient (pred vs true)")
plt.ylabel("Pearson correlation")
plt.tight_layout()
plt.show()