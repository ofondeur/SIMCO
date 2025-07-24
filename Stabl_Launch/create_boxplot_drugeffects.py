import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    drug_to_use_list = ['PRA','LPZ','SALPZ','SA','MF','CHT','THF','RIF','MAP']
    results_path="./Results/comp_9drugs_long_pred_96_tr_vfin/run_/results_no_treatment"

    prediction_csv_path = os.path.join(results_path, 'prediction_drugs_final.csv')
    df = pd.read_csv(prediction_csv_path, index_col=0)

    preterm_patients=['17_G1', '17_G2', '27_G1', '27_G2', '27_G3', '33_G1', '33_G2',
        '33_G3', '3_G1', '3_G2', '3_G3', '5_G1', '5_G2', '5_G3', '8_G1',
        '8_G2', '8_G3']
    preterm_patients_name=['33', '27', '17', '8', '5', '3']
    col_baseline = df.columns[1]
    no_treat_preds = df[col_baseline]

    # Create Delta DataFrame, difference between no treatment and each drug
    diffs = []
    for drug in drug_to_use_list:
        if drug in df.columns:
            delta = no_treat_preds-df[drug]
            for patient, value in delta.items():
                patient_id=df.iloc[patient,0]
                diffs.append({'Drug': drug, 'Patient': patient_id, 'Delta': value,'Preterm': patient_id in preterm_patients})
        else:
            print(f"[WARNING] Drug '{drug}' not found in prediction file.")


    df_deltas = pd.DataFrame(diffs)

    df_deltas['patient_name']=df_deltas['Patient'].str.split('_').str[0]

    df_average_delta = df_deltas.groupby(['patient_name', 'Drug'], as_index=False)['Delta'].mean()
    df_average_delta['Preterm']=df_average_delta['patient_name'].isin(preterm_patients_name)
    df_deltas['Drug'] = pd.Categorical(df_deltas['Drug'], categories=sorted(df_deltas['Drug'].unique()), ordered=True)

    group_means = df_deltas.groupby("Drug")["Delta"].mean()

    print(group_means)

    group_stats = df_deltas.groupby("Drug")["Delta"].agg(Mean="mean", SEM="sem").reset_index()

    colors = ["#102E4D" if val >= 0 else "#E5702D" for val in group_stats["Mean"]]

    # Create barplot to compare drug effects on DOS predictions
    plt.figure(figsize=(5, 8))
    y_pos = np.arange(len(group_stats["Drug"]))

    bars = plt.barh(y_pos, group_stats["Mean"], xerr=group_stats["SEM"], capsize=5, color=colors)

    plt.axvline(0, linestyle="--", color="gray", linewidth=1)
    plt.yticks(y_pos, group_stats["Drug"])
    plt.xlabel("Mean Δ (no drug - with drug)")
    plt.title("Effects of different drugs")
    plt.tight_layout()

    output_path = os.path.join(results_path, "drug_effects_barplot_colored.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Barplot saved under:", output_path)

if __name__ == "__main__":
    main()