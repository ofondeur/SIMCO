import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

drug_to_use_list = ['PRA','LPZ','SALPZ','SA','MF','CHT','THF','RIF']
results_path="/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/comp_8drugs_untreated_unstim/results_no_treatment"

prediction_csv_path = os.path.join(results_path, 'prediction_drugs_final.csv')
df = pd.read_csv(prediction_csv_path, index_col=0)

preterm_patients=['17_G1', '17_G2', '27_G1', '27_G2', '27_G3', '33_G1', '33_G2',
       '33_G3', '3_G1', '3_G2', '3_G3', '5_G1', '5_G2', '5_G3', '8_G1',
       '8_G2', '8_G3']
preterm_patients_name=['33', '27', '17', '8', '5', '3']
col_baseline = df.columns[1]
no_treat_preds = df[col_baseline]


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
plt.figure(figsize=(8, 6))

ax = sns.boxplot(
    x="Drug", y="Delta", data=df_deltas,
    color="lightgray", fliersize=0,
    showmeans=True, meanline=True,
    meanprops={"color": "black", "linestyle": "-", "linewidth": 2},
    medianprops={"visible": False}
)

sns.stripplot(
    x="Drug", y="Delta",
    data=df_deltas[df_deltas["Preterm"] == False],
    color="black", size=4, jitter=True, alpha=0.5
)

sns.stripplot(
    x="Drug", y="Delta",
    data=df_deltas[df_deltas["Preterm"] == True],
    color="red", size=5, jitter=True, alpha=0.8,
    marker='o', edgecolor="darkred", linewidth=0.4
)

for i, drug in enumerate(df_deltas['Drug'].cat.categories):
    med = df_deltas[(df_deltas["Drug"] == drug) & (df_deltas["Preterm"])]['Delta'].mean()
    if not np.isnan(med):
        ax.hlines(
            y=med, xmin=i - 0.2, xmax=i + 0.2,
            colors='blue', linewidth=2.5, label="Preterm Median" if i == 0 else ""
        )

plt.axhline(0, linestyle="--", color="red", linewidth=1)

# Layout
plt.title("Effects of different drugs")
plt.ylabel("Delta (prediction without drug - with drug)")
plt.xticks(rotation=45)
plt.tight_layout()

output_path = os.path.join(results_path, "drug_effects_boxplot_preterm_red.pdf")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
print("Boxplot with preterms in red saved under:", output_path)
