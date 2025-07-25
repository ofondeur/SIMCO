import os
import anndata as ad
from cellot.utils.helpers import load_config
from cellot.models.cellot import load_networks
from cellot.data.cell import read_list
import torch
from pathlib import Path

def load_patient_anndata(data_path, patient, cell_type, drug_prefix,celltype_name_map):
    adata = ad.read_h5ad(data_path, backed='r')
    mask = (
        (adata.obs['stim'] == 'Unstim') &
        (adata.obs['cell_type'] == celltype_name_map[cell_type]) &
        (adata.obs['drug'].str.startswith(drug_prefix)) &
        (adata.obs['sampleID'] == patient)
    )
    
    subset = adata[mask].to_memory()
    adata.file.close()
    return subset

def load_data_networks_stimspred_OOL(result_path, unstim_data_path, stim, model, cell_type, patient,drug_used,celltype_name_map):
    """Load data and model for a specific patient and condition, to predict drug response."""
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
        features_eval = read_list('../datasets/ptb_concatenated_per_condition_celltype/13features.txt')
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
        celltype_name_map=celltype_name_map
    )

    if anndata.shape[0] == 0:
        print(f"[WARN] No data for patient {patient}, stim {stim}, cell {cell_type}", flush=True)
        return

    anndata = anndata[:, features_input].copy()
    return anndata, g, features_eval, features_input, semisuffled_features

def load_data_networks_drug_OOL(result_path, unstim_data_path, stim):
    """Load data and model for a specific patient and condition, to predict drug response."""
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
        features_eval = read_list('../datasets/ptb_concatenated_per_condition_celltype/13features.txt')
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
    
    anndata_to_predict=anndata_to_predict[anndata_to_predict.obs['stim']==stim].copy()
    return anndata_to_predict, g, features_eval, features_input, semisuffled_features