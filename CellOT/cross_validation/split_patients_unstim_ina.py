import os
import anndata as ad
import pandas as pd


UNSTIM_H5_PATH = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ool_by_patient/ool_Unstim_final.h5ad"

PATIENT_LIST_PATH = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/patients_ina.txt"

OUTPUT_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ool_by_patient/unstim_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Loading full unstim AnnData from: {UNSTIM_H5_PATH}")
adata = ad.read_h5ad(UNSTIM_H5_PATH, backed="r")  # Lecture disque uniquement

with open(PATIENT_LIST_PATH, "r") as f:
    patient_ids = [line.strip() for line in f if line.strip()]

print(f"[INFO] Will export unstim data for {len(patient_ids)} patients")

for patient in patient_ids:
    print(f"[INFO] Processing patient {patient}")
    mask = (adata.obs['sampleID'] == patient)
    if mask.sum() == 0:
        print(f"[WARN] No data found for patient {patient}, skipping.")
        continue

    adata_patient = adata[mask].to_memory()  # Charge en mémoire uniquement les cellules du patient
    output_path = os.path.join(OUTPUT_DIR, f"{patient}_Unstim.h5ad")
    adata_patient.write(output_path)
    print(f"[WRITE] Saved to {output_path}")

adata.file.close()

print("[DONE] All patient-specific .h5ad files have been written.")
