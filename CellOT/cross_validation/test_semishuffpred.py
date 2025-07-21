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
drug_used='DMSO'
def batch_correct(data,config,model):
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
                            print('############### entered #######')
                            mask_treat = (data.obs["drug"] == treat)
                            for stim in ["Unstim", config.data.target]:
                                mask_stim = (data.obs[config.data.condition] == stim)
                                mask = mask_patient & mask_pop & mask_treat & mask_stim
                                print(f"[DEBUG] Treat={treat}, Stim={stim}, Patient={patient}, Pop={population}, Marker={marker}")
                                print(f"[DEBUG] Nb obs dans masque = {mask.sum()}")

                                if not mask.any():
                                    continue
    
                                row = df[
                                    (df["Patient"] == patient)
                                    & (df["Treatment"] == treat)
                                    & (df["Stim"] == stim)
                                    & (df["Marker"] == marker)
                                    & (df["Population"] == population)
                                ]
                                print('row', row)
                                if not row.empty:
                                    corrected = row.iloc[0]["correctedMedian"]
                                    real = np.median(data[mask, marker].X)
                                    shift = corrected - real
                                    data[mask, marker].X += shift
                                if row.empty:
                                    print(f"[EMPTY] Patient={patient}, Treatment={treat}, Stim={stim}, Marker={marker}, Pop={population}")

                                print('One correction done')
    
                    else:
                       
                        mask_stim = (data.obs[config.data.condition] == config.data.stim)
                    
                        row = df[
                            (df["Patient"] == patient)
                            & (df["Treatment"] == config.data.drug_used)
                            & (df["Stim"] == config.data.stim)
                            & (df["Marker"] == marker)
                            & (df["Population"] == population)
                        ]

                        if not row.empty:
                            mask_treat=(data.obs["drug"] == config.data.drug_used)
                            mask = mask_patient & mask_pop & mask_treat & mask_stim
                            corrected = row.iloc[0]["correctedMedian"]
                            dmso_target=row.iloc[0]["use_DMSO"]
                            real = np.median(data[mask, marker].X)
                            shift = corrected - real
                            data[mask, marker].X += shift
                            
                        row = df[
                            (df["Patient"] == patient)
                            & (df["Treatment"] == dmso_target)
                            & (df["Stim"] == config.data.stim)
                            & (df["Marker"] == marker)
                            & (df["Population"] == population)
                        ]

                        if not row.empty:
                            mask_treat=(data.obs["drug"] == dmso_target)
                            mask = mask_patient & mask_pop & mask_treat & mask_stim
                            corrected = row.iloc[0]["correctedMedian"]
                            real = np.median(data[mask, marker].X)
                            shift = corrected - real
                            data[mask, marker].X += shift
                            
                            keep_mask = mask_patient & mask_pop & mask_stim & (
                                (data.obs["drug"] == config.data.drug_used) |
                                (data.obs["drug"] == dmso_target)
                            )
                            
                            rename_mask = keep_mask & (data.obs["drug"] == dmso_target)
                            data.obs.loc[rename_mask, "drug"] = "DMSO"
                            
                            data._inplace_subset_obs(keep_mask)
    else:
        print(f'############### no correction {model} ###############')
    return(data)
def predict_from_unstim_data(result_path, unstim_data_path, output_path,stim,test_patients,model):
    config_path = os.path.join(result_path, "config.yaml")
    chkpt = os.path.join(result_path, "cache/model.pt")
    
    feats_input_path= os.path.join(result_path, "features_input_names.txt")
    feats_eval_path= os.path.join(result_path, "features_eval_names.txt")
    semisuffled_features_path= os.path.join(result_path, "semisuffled_features.txt")
    
    # load the config and then the model (f,g)
    config = load_config(config_path)
    if not Path(chkpt).exists():
        print(f"############################## chkpt problem, {chkpt} does not exist #############################################")
    restore=chkpt
    (_, g), _, _ = load(config,restore)
    #(_,g)=load_networks(config)
    #if restore is not None and Path(restore).exists():
        #ckpt = torch.load(restore)
        #g.load_state_dict(ckpt["g_state"])
    g.eval()
    
    # load the data to predict
    anndata_to_predict = ad.read(unstim_data_path)
    
    
    anndata_to_predict=anndata_to_predict[anndata_to_predict.obs['drug'].str.startswith(drug_used)].copy()
    list_drug=anndata_to_predict.obs['drug'].unique()
    print(f'########## andata drug: {list_drug} ###########')
    # load the features used in the training (input, output, eval)
    if os.path.exists(feats_eval_path):
        features_evaluation=read_list(feats_eval_path) # features we want to predict, present in the stim (but not necessarly unstim)
        features_input=read_list(feats_input_path) # features we use as input of our model, present in the unstim (but not necessarly stim)
        semisuffled_features=read_list(semisuffled_features_path) # features we associate to the features_evaluation (e.g. CD25 for CD25 if CD25 is both in eval and input, else a random marker in input), present in the unstim (but not necessarly stim)
    else:
        features_evaluation=read_list('/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/features.txt')
        features_input=features_evaluation
        semisuffled_features=features_evaluation
    features=read_list('/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/all_features.txt')
    
    print(test_patients)
    print(anndata_to_predict.obs['patient'].unique())
    anndata_to_predict = anndata_to_predict[anndata_to_predict.obs['patient'].isin(test_patients)].copy()
    anndata_to_predict=batch_correct(anndata_to_predict,config,model).copy()
    unstim_anndata_to_predict = anndata_to_predict[:, features_input].copy() # filter the input on the markers we want to use to predict
    unstim_anndata_to_predict=unstim_anndata_to_predict[anndata_to_predict.obs['stim']=='Unstim'].copy()
    

    # predict the data (first put it in the dataset format)
    dataset_args = {}
    dataset = AnnDataDataset(unstim_anndata_to_predict.copy(), **dataset_args) #transform the dataset to the expected format
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    inputs = next(iter(loader))
    outputs = g.transport(inputs.requires_grad_(True)).detach().numpy()
    
    predicted = ad.AnnData(
        outputs,
        obs=dataset.adata.obs.copy(),
        var=dataset.adata.var.copy(),
    )
    print('my method before filter', predicted.X, predicted.var_names, np.median(predicted.X, axis=0))
    predicted=predicted[:,semisuffled_features]
    predicted.obs['stim']=stim
    print('var names for prediction',predicted.var_names) 
    predicted.var_names=features_evaluation # rename the prediction markers, so that the markers' name are the same as the true stim data (target)
    original_anndata = anndata_to_predict[anndata_to_predict.obs['stim'] == stim].copy()

    original_anndata.obs["state"] = "true_corrected" # to know if this is the prediction or original data
    
    original_anndata=original_anndata[:,features_evaluation] 
    
    predicted.obs["state"] = "predicted"
    
    concatenated = ad.concat([predicted, original_anndata], axis=0)
    #print(concatenated.obs['drug'].unique(), 'concat')
    #print(predicted.obs['drug'].unique(), 'predicted')
    # save the prediction in the desired format
    #print('True', original_anndata.X, original_anndata.var_names,np.median(original_anndata.X, axis=0))
    #print('my method', predicted.X, predicted.var_names, np.median(predicted.X, axis=0))
    #print('absolute error', abs(np.median(predicted.X, axis=0)-np.median(original_anndata.X, axis=0)))
    #print('MAE', np.mean(abs(np.median(predicted.X, axis=0)-np.median(original_anndata.X, axis=0))))
    return abs(np.median(predicted.X, axis=0)-np.median(original_anndata.X, axis=0))


# read the fold info to have the features and test patients per feature
error_df=pd.DataFrame(columns=['model','CD25', 'HLADR', 'IkB', 'pCREB', 'pERK', 'pMK2', 'pNFkB', 'pS6',
       'pSTAT1', 'pSTAT3', 'pSTAT5', 'pSTAT6', 'pp38', 'PD1', 'CD44', 'PDL1',
       'CD36', 'GLUT1', 'pPLCg', 'pSTAT4'])
for model in [
 'original_med2',
 'original_2',
 'different_IO_med_no_ster2',
 'different_IO_med2',
 'different_IO_mmd2']:
    PTB_ANNDATA_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_to_batchcorrect"
    
    MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}"
    if model=='shuffled_20marks':
        MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_jakob/cross_validation_{model}"
    print(f"[INFO] Beginning predicting")
    for stim in ['IL246']:
        for sanitized_celltype in ['Bcells']:
            for fold_number in [0,1,2,3]:
                if stim=='IL246' and sanitized_celltype=='Bcells':
                    unstim_data_path = f"{PTB_ANNDATA_DIR}/{sanitized_celltype}_HV.h5ad"
                    result_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}"
                    output_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}/pred_CV_{model}test.h5ad"
                    fold_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}"
                    test_patients = ['HV11']
    
    
                    if (os.path.exists(result_path)) and (os.path.exists(fold_path)):
                        fold_model=f"{model}_{fold_number}"
                        error_model=[fold_model]
                        markers_pred=predict_from_unstim_data(result_path, unstim_data_path, output_path, stim,test_patients,model).tolist()
                        error_model=error_model+markers_pred
                        print(len(error_model))
                        
                        error_df.loc[fold_model] = error_model
    print(f"finished predicting for {model}")
error_df.to_csv(f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/errors_4folds_v2_2.csv", index=False)

    