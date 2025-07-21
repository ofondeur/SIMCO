import os
import anndata as ad
from cellot.utils.helpers import load_config
from cellot.utils.loaders import load
from cellot.data.cell import read_list
from cellot.data.cell import AnnDataDataset
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path
import sys

drug_used='DMSO'

def predict_from_unstim_data(result_path, unstim_data_path, output_path,stim,test_patients):
    config_path = os.path.join(result_path, "config.yaml")
    chkpt = os.path.join(result_path, "cache/model.pt")
    
    suffled_features_path= os.path.join(result_path, "suffled_features.txt")
    
    # load the config and then the model (f,g)
    print(config_path)
    config = load_config(config_path)
    if not Path(chkpt).exists():
        print(f'############################## chkpt problem, {chkpt} does not exist #############################################')
    
    (_, g), _, _ = load(config, restore=chkpt)
    g.eval()
    
    # load the data to predict
    anndata_to_predict = ad.read(unstim_data_path)
    anndata_to_predict=anndata_to_predict[anndata_to_predict.obs['drug']==drug_used].copy()
    
    # load the features used in the training
    suffled_features=read_list(suffled_features_path) # features we associate to the features_evaluation (e.g. CD25 for CD25 if CD25 is both in eval and input, else a random marker in input), present in the unstim (but not necessarly stim)
    features_evaluation= read_list(config.data.features)
    features=read_list(config.data.features)
    anndata_to_predict = anndata_to_predict[:, features].copy()
    print(test_patients)
    
    anndata_to_predict = anndata_to_predict[anndata_to_predict.obs['patient'].isin(test_patients)].copy()
    
    unstim_anndata_to_predict = anndata_to_predict.copy() # filter the input on the markers we want to use to predict
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
    predicted=predicted[:,suffled_features]
    predicted.obs['stim']=stim
    
    predicted.var_names=features_evaluation # rename the prediction markers, so that the markers' name are the same as the true stim data (target)
    original_anndata = anndata_to_predict[anndata_to_predict.obs['stim'] == stim].copy()

    original_anndata.obs["state"] = "true_corrected" # to know if this is the prediction or original data
    
    original_anndata=original_anndata[:,features_evaluation] 
    
    predicted.obs["state"] = "predicted"
    
    concatenated = ad.concat([predicted, original_anndata], axis=0)
    print(concatenated.obs['drug'].unique(), 'concat')
    print(predicted.obs['drug'].unique(), 'predicted')
    # save the prediction in the desired format
    if output_path.endswith(".csv"):
        concatenated = concatenated.to_df()
        concatenated.to_csv(output_path)

    elif output_path.endswith(".h5ad"):
        print(output_path)
        concatenated.write(output_path)
    return

model='shuffled_20marks' # choose among 'olivier', 'peter', 'original','all_markers' ,'39m_OG'

# read the fold info to have the features and test patients per feature

FOLD_INFO_FILE = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info_{model}.csv"
fold_info = defaultdict(lambda: defaultdict(dict)) # Structure: fold_info[stim][sanitized_celltype][fold_index] = [test_patient1, ...]
stim_celltype_pairs_in_folds = set() # Keep track of (stim, original_celltype) pairs processed

try:
    with open(FOLD_INFO_FILE, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        lines = lines[1:][::-1]  
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 4: continue # stim, sanitized, original, fold, patient1...
            stim, sanitized_ct, original_ct, fold_idx_str = parts[:4]
            test_patients = parts[4:]
            try:
                fold_idx = int(fold_idx_str)
                fold_info[stim][sanitized_ct][fold_idx] = test_patients
                stim_celltype_pairs_in_folds.add((stim, original_ct))
            except ValueError:
                print(f"[WARN] Invalid fold index '{fold_idx_str}' in line: {line.strip()}")
    print(f"Loaded fold info for {len(fold_info)} stims.")
    print(f"Total (stim, original_celltype) pairs with fold info: {len(stim_celltype_pairs_in_folds)}")
    
except FileNotFoundError:
    sys.exit(f"FATAL ERROR: Fold info file not found: {FOLD_INFO_FILE}")
except Exception as e:
    sys.exit(f"FATAL ERROR: Failed to read fold info file: {e}")


PTB_ANNDATA_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype"
MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results_jakob/cross_validation_{model}"

print(f"[INFO] Beginning predicting")

for stim in fold_info:
    for sanitized_celltype in fold_info[stim]:
        for fold_number in fold_info[stim][sanitized_celltype]:
        
            test_patients = fold_info[stim][sanitized_celltype][fold_number]

            unstim_data_path = f"{PTB_ANNDATA_DIR}/{sanitized_celltype}_HV.h5ad"
            result_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}"
            output_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}/pred_CV_{model}.h5ad"
            fold_path = f"{MODEL_BASE_DIR}/{stim}/{sanitized_celltype}/model-{stim}_{sanitized_celltype}_fold{fold_number}"
            
            if (not os.path.exists(output_path)) and (os.path.exists(result_path)) and (os.path.exists(fold_path)):
                predict_from_unstim_data(result_path, unstim_data_path, output_path, stim,test_patients)
                print('output_path')
        print(f"[INFO] Predicted {sanitized_celltype} for {stim}")
        
print("finished predicting")