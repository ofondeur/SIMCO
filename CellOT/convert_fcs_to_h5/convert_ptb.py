import pandas as pd
import anndata as ad
from FlowCytometryTools import FCMeasurement
import numpy as np
import os
def arcsinh_transform(X,cofactor=5):
    return np.arcsinh(X/cofactor)
    
def extract_patient_from_path(file_path):
    """extract patient id from file name (works for DOMS, to adapt for others)"""
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    patient_id = file_name_without_ext.split("_")[1]
    return patient_id
    
def load_data_fcs(path,condition_name,cell_type=None):
    data = FCMeasurement(ID="Sample", datafile=path)
    anndata = ad.AnnData(data.data)
    anndata.obs_names_make_unique()
    anndata.obs['drug']= condition_name
    
    if cell_type:
        anndata.obs['cell_type']= cell_type
    patient_id = extract_patient_from_path(path)
    anndata.obs['patient'] = patient_id
    anndata.X=arcsinh_transform(anndata.X)
    return anndata
    
    
def concatenate_2conditions_multiple_data(path_stim_list,stim_name,path_unstim_list,unstim_name,outdir_path):
    comb_unstim_anndata = load_data_fcs(path_unstim_list[0][0],unstim_name,path_unstim_list[0][1])
    comb_stim_anndata = load_data_fcs(path_stim_list[0][0],stim_name,path_stim_list[0][1])

    for path_stim in path_stim_list[1:]:
        if path_stim[1]:
            stim_anndata = load_data_fcs(path_stim[0],stim_name,path_stim[1])
        else:
            stim_anndata = load_data_fcs(path_stim[0],stim_name)
        assert isinstance(comb_stim_anndata, ad.AnnData), f"Expected AnnData, got {type(comb_stim_anndata)}"
        assert isinstance(stim_anndata, ad.AnnData), f"Expected AnnData, got {type(stim_anndata)}"
        comb_stim_anndata.obs.reset_index(drop=True, inplace=True)
        stim_anndata.obs.reset_index(drop=True, inplace=True)

        comb_stim_anndata = ad.concat([comb_stim_anndata, stim_anndata], join='outer', axis=0)
        
    for path_unstim in path_unstim_list[1:]:
        if path_unstim[1]:
            unstim_anndata = load_data_fcs(path_unstim[0],unstim_name,path_unstim[1])
        else:
            unstim_anndata = load_data_fcs(path_unstim[0],unstim_name)
        comb_unstim_anndata = ad.concat([comb_unstim_anndata, unstim_anndata], join='outer', axis=0)
        
    comb_stim_anndata.var = comb_stim_anndata.var.astype(str)
    comb_unstim_anndata.var = comb_unstim_anndata.var.astype(str)

    combined_anndata = ad.concat([comb_unstim_anndata, comb_stim_anndata], join='outer', axis=0)
    
    directory=os.path.dirname(outdir_path)
    os.makedirs(directory, exist_ok=True)
    combined_anndata.write_h5ad(outdir_path)

    return 
    
def create_list_of_paths2(
    directory, stimulation, cell_type=None, sample=None, patient_included=[]
):
    paths_list = []
    for filename in os.listdir(directory):
        if len(patient_included) > 0:
            for patient in patient_included:
                if (
                    stimulation in filename
                    and (cell_type is None or cell_type in filename)
                    and (sample is None or sample in filename)
                    and patient in filename
                ):
                    paths_list.append([os.path.join(directory, filename),cell_type])
        else:
            if (
                stimulation in filename
                and (cell_type is None or cell_type in filename)
                and (sample is None or sample in filename)
            ):
                paths_list.append([os.path.join(directory, filename),cell_type])
    return paths_list

ptb_stim_list=['TNFa','LPS','IFNa','PI','GMCSF','IL33','IL246']
ptb_cell_list=['Bcells', 'CD4Tcm.', 'CD4Tcm CCR2+', 'CD4Teff', 'CD4Tem.',
       'CD4Tem CCR2+', 'CD4Tnaive', 'CD4Tregs', 'CD4negCD8negTcells',
       'CD56hiCD16negNK', 'CD56loCD16posNK', 'CD8Tcells Th1', 'CD8Tcm.',
       'CD8Tcm CCR2+', 'CD8Teff', 'CD8Tem.', 'CD8Tem CCR2+', 'CD8Tnaive',
       'Granulocytes', 'MDSCs', 'NK cells CD11c+', 'NK cells CD11c-',
       'NKT', 'cMCs', 'intMCs', 'mDCs', 'ncMCs', 'pDCs']
states=['HV','PV']
c=0
tot=len(ptb_stim_list)*len(ptb_cell_list)
for state in states:
    if state=='PV':
         nb_file==11*12
    else:
         nb_file=8*12
    for cell_type in ptb_cell_list:
        for stim in ptb_stim_list:
            path_unstim_list=create_list_of_paths2(directory='./cells_combined/raw_data/ptb_final_rawdata',stimulation='Unstim',cell_type=cell_type,sample=state)
            path_stim_list=create_list_of_paths2(directory='./cells_combined/raw_data/ptb_final_rawdata',stimulation=stim,cell_type=cell_type,sample=state)
            assert len(path_unstim_list) == nb_file, f'{len(path_unstim_list)} file found for {cell_type} and {stim}'
            assert len(path_stim_list) == nb_file, f'{len(path_unstim_list)} for {cell_type} and {stim}'
            if cell_type.endswith('.'):
                cell_type=cell_type[:,-1]
            res=concatenate_2conditions_multiple_data(path_stim_list,stim,path_unstim_list,'Unstim',"./datasets/PTB_just_concat/PTB_concatenated_"+state+'_'+stim.replace(' ','_')+"_"+cell_type.replace(' ','_')+".h5ad")
            c+=1
            print('done ',c,'/',tot)
