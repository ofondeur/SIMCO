import os
import pandas as pd
import numpy as np
import anndata as ad
from tqdm import tqdm

# Configuration
CSV_PATH = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/ina_13OG_final_long.csv"
UNSTIM_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ool_by_patient/unstim_final"
OUTPUT_CSV = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/ina_13OG_final_long_allstims.csv"  # peut être remplacé par un nouveau chemin si tu veux garder l'original
STIM_NAME = "Unstim"

# Mapping celltype nom simple → nom utilisé dans AnnData
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

print(f"[INFO] Loading base CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)


# Colonnes ajoutées
new_data = {}
path_patients_ina='/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/patients_ina.txt'
with open(path_patients_ina, 'r') as f:
    patients_ina_list = [line.strip() for line in f if line.strip()]
    
for patient in tqdm(patients_ina_list, desc="Processing patients"):
    h5_path = os.path.join(UNSTIM_DIR, f"{patient}_Unstim.h5ad")
    if not os.path.exists(h5_path):
        print(f"[WARN] Missing file for patient {patient}")
        continue

    adata = ad.read_h5ad(h5_path)
    adata = adata[adata.obs['stim'] == STIM_NAME]

    for simple_ct, full_ct in celltype_name_map.items():
        mask = adata.obs['cell_type'] == full_ct
        if not np.any(mask):
            print('no mask')
            continue
        sub = adata[mask].copy()
        for i, marker in enumerate(sub.var_names):
            colname = f"{simple_ct}_{marker}_{STIM_NAME}"
            median_val = np.nanmedian(sub[:, i].X)
            df.loc[df["Individual"] == patient, colname] = median_val
    adata.file.close()

# Fusion avec le DataFrame existant
df_new = pd.DataFrame.from_dict(new_data, orient="index")
df_combined = df.join(df_new, how="left")

# Sauvegarde
df_combined.reset_index(inplace=True)
df_combined.to_csv(OUTPUT_CSV, index=False)
print(f"[SUCCESS] Updated CSV saved to: {OUTPUT_CSV}")
