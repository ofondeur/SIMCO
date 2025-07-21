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
from matplotlib.ticker import MaxNLocator
import torch
from pathlib import Path
import sys
from collections import defaultdict
drug_used = 'DMSO'
model = 'original1'

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

def load_patient_anndata_test(data_path, patient, cell_type, drug_prefix,stim):
    adata = ad.read_h5ad(data_path, backed='r')
    mask = (
        (adata.obs['stim'].isin(['Unstim',stim])) &
        (adata.obs['drug'].str.startswith(drug_prefix)) &
        (adata.obs['patient'] == patient)
    )
    
    subset = adata[mask].to_memory()
    adata.file.close()
    return subset
    
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

def create_density_plots(dist_data, out_file, title_suffix=""):
    sns.set_theme(style="white")  # No grid

    pts_sorted = sorted(dist_data.keys())
    num_plots = len(pts_sorted)
    cols = min(3, num_plots)
    rows = int(np.ceil(num_plots / cols))

    fig_size = max(5 * max(cols, rows), 8)
    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_size,fig_size), constrained_layout=True
    )
    if num_plots == 1:
        axes = np.array([axes])

    cat_labels = ["Unstim", "Stim True", "Stim Pred"]
    cat_colors = ["#4e4e50", "#76c7c0", "#c1443c"]

    for i, (pt, ax) in enumerate(zip(pts_sorted, axes.flatten())):
        for label, color in zip(cat_labels, cat_colors):
            arr = dist_data[pt][label]
            
            if arr.size > 0:
                sns.kdeplot(
                    arr,
                    ax=ax,
                    #label=f"{label} (n={arr.size})",
                    color=color,
                    fill=False,
                    linewidth=7,
                    alpha=0.9,
                )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        ax.tick_params(
            axis='both',
            which='both',
            direction='inout',
            length=4,
            width=6,
            color='blue',
            labelsize=10)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(1.5)
            ax.spines[spine].set_color("black")

        ax.set_facecolor("white")

    for j in range(i + 1, len(axes.flatten())):
        fig.delaxes(axes.flatten()[j])

    plt.savefig(out_file + ".pdf", format="pdf", bbox_inches="tight")

    plt.close()



def plot_result(data_path, outdir_path,stim,model,cell_type,patient, marker):
    target = load_patient_anndata_test(data_path, patient, cell_type, drug_used,stim)
    # for marker in ['CD25', 'HLADR','IkB','pCREB','pERK','pMK2','pNFkB','pS6','pSTAT1','pSTAT3','pSTAT5','pSTAT6','pp38']:
    target1 = target[:, marker].copy()
    stim_serie = pd.Series(
        target1[(target1.obs["stim"] == stim)].X.flatten(), name="Stim True"
    )
    unstim = pd.Series(
        target1[target1.obs["stim"] == "Unstim"].X.flatten(), name="Unstim"
    )
    prediction=predict_per_patient(result_path, data_path, stim, model, cell_type, patient)
    prediction=prediction[:,marker].copy()
    pred = pd.Series(
        prediction.X.flatten(), name="Stim Pred"
    )
    print('loaded')
    dist_data = {
        "Patient_1": {
            "Stim True": stim_serie.values,
            "Stim Pred": pred,
            "Unstim": unstim.values,
        }
    }
    output_path=os.path.join(outdir_path,f"{stim}_{cell_type}_{marker}")
    create_density_plots(dist_data, output_path, title_suffix="")
    print(f"saved under {output_path}")
    return
MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}"
FOLD_INFO_FILE = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info_original_20marks.csv"
fold_info = defaultdict(lambda: defaultdict(dict))

with open(FOLD_INFO_FILE, 'r') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split(',')
        if len(parts) < 5:
            continue
        stim, sanitized_ct, original_ct, fold_idx, *patients = parts
        fold_info[stim][sanitized_ct][int(fold_idx)] = patients

jobs=[['CD4Tregs', 'pSTAT5'],['CD8Tnaive', 'pNFkB'],['CD4Tnaive', 'pSTAT6'],['cMCs', 'pSTAT3'],['pDCs', 'pSTAT6']]
stim='IL246'
for job in jobs:
    fold_idx='0'
    cell_type,marker=job[0],job[1]                                                           #  fold_info[stim][sanitized_ct]
    patient=fold_info[stim][sanitized_ct][int(fold_idx)][0]

    data_path = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/{cell_type}_HV.h5ad"
    result_path = f"{MODEL_BASE_DIR}/{stim}/{cell_type}/model-{stim}_{cell_type}_fold{int(fold_idx)}"
    model_file = os.path.join(result_path, "cache/model.pt")
    outdir_path=f"{MODEL_BASE_DIR}/plots"
    os.makedirs(outdir_path, exist_ok=True)
    plot_result(data_path, outdir_path,stim,model,cell_type,patient,marker)