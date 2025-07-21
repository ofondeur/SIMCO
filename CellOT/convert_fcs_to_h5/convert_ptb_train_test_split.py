from FlowCytometryTools import FCMeasurement
import numpy as np
import pandas as pd
import anndata as ad
import os
def arcsinh_transform(X,cofactor=5):
    return np.arcsinh(X/cofactor)
    
def extract_patient_from_path(file_path):
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

        # Concaténer les deux objets AnnData
        comb_stim_anndata = ad.concat([comb_stim_anndata, stim_anndata], join='outer', axis=0)
        #comb_stim_anndata = comb_stim_anndata.concatenate(stim_anndata)
        
    for path_unstim in path_unstim_list[1:]:
        if path_unstim[1]:
            unstim_anndata = load_data_fcs(path_unstim[0],unstim_name,path_unstim[1])
        else:
            unstim_anndata = load_data_fcs(path_unstim[0],unstim_name)
        comb_unstim_anndata = ad.concat([comb_unstim_anndata, unstim_anndata], join='outer', axis=0)
        #comb_unstim_anndata = comb_unstim_anndata.concatenate(unstim_anndata)
    comb_stim_anndata.var = comb_stim_anndata.var.astype(str)
    comb_unstim_anndata.var = comb_unstim_anndata.var.astype(str)

    combined_anndata = ad.concat([comb_unstim_anndata, comb_stim_anndata], join='outer', axis=0)
    #combined_anndata.write(outdir_path)
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
#perio_cell_list=['Granulocytes (CD45-CD66+)','B-Cells (CD19+CD3-)','Classical Monocytes (CD14+CD16-)','MDSCs (lin-CD11b-CD14+HLADRlo)','mDCs (CD11c+HLADR+)','pDCs(CD123+HLADR+)','Intermediate Monocytes (CD14+CD16+)','Non-classical Monocytes (CD14-CD16+)','CD56+CD16- NK Cells','CD56loCD16+NK Cells','NK Cells (CD7+)','CD4 T-Cells','Tregs (CD25+FoxP3+)','CD8 T-Cells','CD8-CD4- T-Cells']
ptb_cell_list=['Bcells', 'CD4Tcm.', 'CD4Tcm CCR2+', 'CD4Teff', 'CD4Tem.',
       'CD4Tem CCR2+', 'CD4Tnaive', 'CD4Tregs', 'CD4negCD8negTcells',
       'CD56hiCD16negNK', 'CD56loCD16posNK', 'CD8Tcells Th1', 'CD8Tcm.',
       'CD8Tcm CCR2+', 'CD8Teff', 'CD8Tem.', 'CD8Tem CCR2+', 'CD8Tnaive',
       'Granulocytes', 'MDSCs', 'NK cells CD11c+', 'NK cells CD11c-',
       'NKT', 'cMCs', 'intMCs', 'mDCs', 'ncMCs', 'pDCs']
states=['HV']
patient_train=['HV04','HV05','HV06','HV07','HV08','HV09']
patient_test=['HV10','HV11']
c=0
tot=len(ptb_stim_list)*len(ptb_cell_list)
for state in states:
    if state=='PV':
         nb_file==11*12
    else:
         nb_file=8*12
    for cell_type in ptb_cell_list:
        for stim in ptb_stim_list:
            state='DMSO'
            path_unstim_list=create_list_of_paths2(directory='./cells_combined/raw_data/ptb_final_rawdata',stimulation='Unstim',cell_type=cell_type,sample=state,patient_included=patient_train)
            path_stim_list=create_list_of_paths2(directory='./cells_combined/raw_data/ptb_final_rawdata',stimulation=stim,cell_type=cell_type,sample=state,patient_included=patient_train)
            assert len(path_unstim_list) == 18, f'{len(path_unstim_list)} file found for {cell_type} and {stim}'
            assert len(path_stim_list) == 18, f'{len(path_unstim_list)} for {cell_type} and {stim}'
            if cell_type.endswith('.'):
                cell_type_folder=cell_type[:-1]
            else:
                cell_type_folder=cell_type
            res=concatenate_2conditions_multiple_data(path_stim_list,stim,path_unstim_list,'Unstim',"./datasets/PTB_just_concat_train/PTB_concatenated_"+state+'_'+stim.replace(' ','_')+"_"+cell_type_folder.replace(' ','_')+".h5ad")
            
            path_unstim_list=create_list_of_paths2(directory='./cells_combined/raw_data/ptb_final_rawdata',stimulation='Unstim',cell_type=cell_type,sample=state,patient_included=patient_test)
            path_stim_list=create_list_of_paths2(directory='./cells_combined/raw_data/ptb_final_rawdata',stimulation=stim,cell_type=cell_type,sample=state,patient_included=patient_test)
            assert len(path_unstim_list) == 6, f'{len(path_unstim_list)} file found for {cell_type} and {stim}'
            assert len(path_stim_list) == 6, f'{len(path_unstim_list)} for {cell_type} and {stim}'
            res=concatenate_2conditions_multiple_data(path_stim_list,stim,path_unstim_list,'Unstim',"./datasets/PTB_just_concat_test/PTB_concatenated_"+state+'_'+stim.replace(' ','_')+"_"+cell_type_folder.replace(' ','_')+".h5ad")
            
            c+=1
            print('done ',c,'/',tot)