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
from scipy import sparse
import pandas as pd
import numpy as np
drug_used='PRA'
dmso_target='DMSO2'
def batch_correct(data,config,model):
    if sparse.issparse(data.X):
        print('ohhhhhhhh sparse')
        data.X = data.X.toarray()
    if "batch_correction" in config.data:
        df = pd.read_csv(config.data.batch_correction)
    
        present_patients = data.obs["patient"].unique()
    
        
        df = df[df["Patient"].isin(present_patients)]
    
        
        markers = list(set(df["Marker"]).intersection(data.var_names))
        print(f"Correcting markers: {markers}")
        
        for patient in present_patients:
            for population in data.obs["cell_type"].unique():
                # Sous-ensemble Anndata correspondant
                mask_patient = (data.obs["patient"] == patient)
                mask_pop = (data.obs["cell_type"] == population)
    
                for marker in markers:
                    if "drug_used" not in config.data:
                        raise ValueError("No drug specified, need one to batch correct")
                    
                    if config.data.drug_used == "DMSO":
                        for treat in ['DMSO1','DMSO2','DMSO3']:
                            mask_treat = (data.obs["drug"] == treat)
                            for stim in ["Unstim", config.data.target]:
                                mask_stim = (data.obs[config.data.condition] == stim)
                                mask = mask_patient & mask_pop & mask_treat & mask_stim
    
                                if not mask.any():
                                    print('mask seems pretty empty')
                                    continue
    
                                row = df[
                                    (df["Patient"] == patient)
                                    & (df["Treatment"] == treat)
                                    & (df["Stim"] == stim)
                                    & (df["Marker"] == marker)
                                    & (df["Population"] == population)
                                ]
                                if row.empty:
                                    print(f"[EMPTY] Patient={patient}, Treatment={treat}, Stim={stim}, Marker={marker}, Pop={population}")
                                if not row.empty:
                                    corrected = row.iloc[0]["correctedMedian"]
                                    real = np.median(data[mask, marker].X)
                                    shift = corrected - real
                                    data[mask, marker].X += shift
                                    new_median = np.median(data[mask, marker].X)
                                    if not np.isclose(new_median, corrected, atol=1e-6):
                                        print(f"[WARNING] Correction mismatch: expected {corrected}, got {new_median}")
                                    else:
                                        print(f"[OK] Corrected median matches: {corrected}")
                                
    
                    else:
                        
                        mask_stim = (data.obs["stim"] == config.data.stim)
                        drug_used= data.obs['drug'].unique()
                        row1 = df[
                            (df["Patient"] == patient)
                            & (df["Treatment"] == config.data.drug_used)
                            & (df["Stim"] == config.data.stim)
                            & (df["Marker"] == marker)
                            & (df["Population"] == population)
                        ]
                        if not row1.empty:
                            mask_stim = (data.obs["stim"] == config.data.stim)
                            mask_treat = (data.obs["drug"] == config.data.drug_used)
                            mask = mask_patient & mask_pop & mask_treat & mask_stim
                            corrected = row1.iloc[0]["correctedMedian"]
                            dmso_target_feat = row1.iloc[0]["use_DMSO"]
                            if dmso_target_feat != dmso_target:
                                continue
                            real = np.median(data[mask, marker].X)
                            shift = corrected - real
                            data.X[mask.to_numpy(), data.var_names.get_loc(marker)] += shift
                            
                        row = df[
                            (df["Patient"] == patient)
                            & (df["Treatment"] == dmso_target)
                            & (df["Stim"] == config.data.stim)
                            & (df["Marker"] == marker)
                            & (df["Population"] == population)
                        ]
                        if not row.empty:
                            mask_treat = (data.obs["drug"] == dmso_target)
                            mask = mask_patient & mask_pop & mask_treat & mask_stim
                            corrected = row.iloc[0]["correctedMedian"]
                            real = np.median(data[mask, marker].X)
                            shift = corrected - real
                            data.X[mask.to_numpy(), data.var_names.get_loc(marker)] += shift
                            new_median = np.median(data[mask, marker].X)
                            if not np.isclose(new_median, corrected, atol=1e-6):
                                print(f"[WARNING] Correction mismatch: expected {corrected}, got {new_median}")
                            else:
                                print(f"[OK] Corrected median matches: {corrected}")
    
    else:
        print(f'############### no correction {model} ###############')
    
    return(data)
def predict_from_unstim_data(result_path, unstim_data_path, output_path,stim,test_patients,model):
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
    anndata_to_predict=anndata_to_predict[anndata_to_predict.obs['stim']==stim].copy()
    anndata_to_predict = anndata_to_predict[anndata_to_predict.obs['patient'].isin(test_patients)].copy()

    anndata_to_predict=batch_correct(anndata_to_predict,config,model).copy()

    untreated_anndata_to_predict = anndata_to_predict[:, features_input].copy() # filter the input on the markers we want to use to predict
    
    untreated_anndata_to_predict=untreated_anndata_to_predict[untreated_anndata_to_predict.obs['drug']=='DMSO2'].copy()
    
    stims_in_data=untreated_anndata_to_predict.obs['stim'].unique().tolist()
    drugs_in_data=untreated_anndata_to_predict.obs['drug'].unique().tolist()
    cell_type_in_data=untreated_anndata_to_predict.obs['cell_type'].unique().tolist()
    
    print(f"The categories of the anndata before the source, target split is: stim {stims_in_data}, drugs {drugs_in_data}, cells {cell_type_in_data}, with a shape of {untreated_anndata_to_predict.X.shape}")
    print('True untreated', np.median(untreated_anndata_to_predict.X, axis=0))
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
    print('Prediction treatment', np.median(predicted.X, axis=0))
    
    predicted=predicted[:,semisuffled_features]
    predicted.obs['drug']=drug_used
    print('var names for prediction',predicted.var_names) 
    predicted.var_names=features_eval # rename the prediction markers, so that the markers' name are the same as the true stim data (target)
    original_anndata = anndata_to_predict[anndata_to_predict.obs['drug'] == drug_used].copy()

    original_anndata.obs["state"] = "true_corrected" # to know if this is the prediction or original data
    
    original_anndata=original_anndata[:,features_eval] 
    
    predicted.obs["state"] = "predicted"
    print('True', np.median(original_anndata.X, axis=0))
    concatenated = ad.concat([predicted, original_anndata], axis=0)
    
    feat13=read_list('/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/13features.txt')
    predicted=predicted[:,feat13]
    original_anndata=original_anndata[:,feat13]
    print(abs(np.median(predicted.X, axis=0)-np.median(original_anndata.X, axis=0)))
    return abs(np.median(predicted.X, axis=0)-np.median(original_anndata.X, axis=0))


# read the fold info to have the features and test patients per feature
error_df=pd.DataFrame(columns=['model','CD25', 'HLADR', 'IkB', 'pCREB', 'pERK', 'pMK2', 'pNFkB', 'pS6',
       'pSTAT1', 'pSTAT3', 'pSTAT5', 'pSTAT6', 'pp38'])
for model in ['drug_OG_13_HVPV_newBC2','drug_OG_20_HVPV','drug_diffIO_HVPV']:
    PTB_ANNDATA_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_batchcorrected"
    
    MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}"
    if model=='shuffled_20marks':
        MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_jakob/cross_validation_{model}"
    print(f"[INFO] Beginning predicting")
    for stim in ['GMCSF']:
        for sanitized_celltype in ['mDCs']:
            for fold_number in [0,1,2,3]:
                if stim=='GMCSF' and sanitized_celltype=='mDCs':
                    unstim_data_path = f"{PTB_ANNDATA_DIR}/{sanitized_celltype}_HVPV.h5ad"
                    result_path = f"{MODEL_BASE_DIR}/{drug_used}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}"
                    output_path = f"{MODEL_BASE_DIR}/{drug_used}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}/pred_CV_{model}test.h5ad"
                    fold_path = f"{MODEL_BASE_DIR}/{drug_used}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}"
                    test_patients = ['HV10','PV01']
    
    
                    if (os.path.exists(result_path)) and (os.path.exists(fold_path)):
                        fold_model=f"{model}_{fold_number}"
                        error_model=[fold_model]
                        markers_pred=predict_from_unstim_data(result_path, unstim_data_path, output_path, stim,test_patients,model).tolist()
                        error_model=error_model+markers_pred
                        print(len(error_model))
                        
                        error_df.loc[fold_model] = error_model
    print(f"finished predicting for {model}")
error_df.to_csv(f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/error_drug_3model.csv", index=False)

    