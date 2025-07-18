import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, r2_score
from glob import glob

# === Config ===
root_path = "/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/comp_9drugs_long"
clinical_info_path = "/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/Preprocessed_OOL_Clinical.csv"
drug_list = ['PRA','LPZ','SALPZ','SA','MF','CHT','THF','RIF','MAP']
preterm_patients = ['17_G1', '17_G2', '27_G1', '27_G2', '27_G3', '33_G1', '33_G2',
       '33_G3', '3_G1', '3_G2', '3_G3', '5_G1', '5_G2', '5_G3', '8_G1',
       '8_G2', '8_G3']
preterm_patients_name = ['33', '27', '17', '8', '5', '3']


info = pd.read_csv(clinical_info_path)
pre_term_ids = info[info["Delivery"] == "preterm"]["ID"].values
info.set_index("ID", inplace=True)

results = []

for run_dir in sorted(glob(os.path.join(root_path, "run_*"))):
    run_id = os.path.basename(run_dir)

    try:
        prediction_path = os.path.join(run_dir, "results_no_treatment", "prediction_drugs_final.csv")
        if not os.path.exists(prediction_path):
            print(f"[WARNING] Missing predictions in {run_dir}")
            continue

        df = pd.read_csv(prediction_path, index_col=0)
        baseline_col = df.columns[1]
        no_treat_preds = df[baseline_col]

        diffs = []
        for drug in drug_list:
            if drug in df.columns:
                delta = no_treat_preds - df[drug]
                for patient, value in delta.items():
                    patient_id = df.iloc[patient, 0]
                    diffs.append({
                        "Drug": drug,
                        "Patient": patient_id,
                        "Delta": value,
                        "Preterm": patient_id in preterm_patients
                    })

        df_deltas = pd.DataFrame(diffs)
        df_deltas["patient_name"] = df_deltas["Patient"].str.split("_").str[0]
        group_means = df_deltas.groupby("Drug")["Delta"].mean().sort_values(ascending=False)

        if "SALPZ" not in group_means.index:
            print(f"[WARNING] SALPZ not found in {run_dir}")
            continue

        salpz_rank = group_means.index.tolist().index("SALPZ") + 1
        nb_positive_deltas = (group_means > 0).sum()

        preds_path = os.path.join(run_dir, "results_no_treatment", "Training CV", "No_treat", "STABL ALasso", "STABL ALasso predictions.csv")
        if not os.path.exists(preds_path):
            print(f"[WARNING] Missing AUROC preds in {run_dir}")
            continue

        data = pd.read_csv(preds_path)
        data["is_preterm"] = data["patient_id"].apply(lambda x: 1 if x in pre_term_ids else 0)
        data.set_index("patient_id", inplace=True)
        common_ids = info.index.intersection(data.index)

        estimated_duration = info.loc[common_ids, "EGA"] * 7 - data.loc[common_ids].iloc[:, 0]
        y_true = data.loc[common_ids, "is_preterm"].values
        y_pred = estimated_duration.values

        fpr, tpr, _ = roc_curve(y_true, -y_pred)
        roc_auc = auc(fpr, tpr)
        
        # RMSE for all patients
        common_idx = info.index.intersection(data.index)
        rmse_all = np.sqrt(mean_squared_error(
            info.loc[common_idx, 'DOS'],
            data.loc[common_idx, data.columns[0]]
        ))
        
        correlation_pearson = info['DOS'].corr(data.iloc[:,0], method='pearson')

        results.append({
            "Run": run_id,
            "SALPZ_Rank": salpz_rank,
            "Nb_Positive_Deltas": nb_positive_deltas,
            "AUROC": roc_auc,
            "RMSE global": rmse_all,
            "Pearson R": correlation_pearson
        })

    except Exception as e:
        print(f"[ERROR] Failed for {run_dir}: {e}")


out_df = pd.DataFrame(results)
out_df.to_csv(os.path.join(root_path, "gridsearch_summary_2.csv"), index=False)
