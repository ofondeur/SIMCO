import anndata as ad
import os

output_dir='/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype'
os.makedirs(output_dir, exist_ok=True)
data_path='/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype'
cells=['Bcells', 'CD4Tcm', 'CD4Tcm_CCR2pos', 'CD4Teff', 'CD4Tem', 'CD4Tem_CCR2pos', 'CD4Tnaive', 'CD4Tregs', 'CD4negCD8negTcells', 'CD56hiCD16negNK', 'CD56loCD16posNK', 'CD8Tcells_Th1', 'CD8Tcm', 'CD8Tcm_CCR2pos', 'CD8Teff', 'CD8Tem', 'CD8Tem_CCR2pos', 'CD8Tnaive', 'Granulocytes', 'MDSCs', 'NK_cells_CD11cpos', 'NK_cells_CD11cneg', 'NKT', 'cMCs', 'intMCs', 'mDCs', 'ncMCs', 'pDCs']

for sanitized_cell in cells:
    HV_path=f"{data_path}/{sanitized_cell}_HV.h5ad"
    PV_path=f"{data_path}/{sanitized_cell}_PV.h5ad"
    outdir_path=f"{output_dir}/{sanitized_cell}_HVPV.h5ad"
    if not os.path.exists(HV_path):
        print(f"Missing file: {HV_path}")
        continue
    if not os.path.exists(PV_path):
        print(f"Missing file: {PV_path}")
        continue
        
    HV_anndata=ad.read_h5ad(HV_path)
    PV_anndata=ad.read_h5ad(PV_path)
    
    combined_condition_data = ad.concat([HV_anndata, PV_anndata], join='outer', axis=0)
    combined_condition_data.write_h5ad(outdir_path)
    