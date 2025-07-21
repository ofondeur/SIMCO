import os
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_density_plots(dist_data, out_file, title_suffix=""):
    sns.set_theme(style="whitegrid")
    pts_sorted = sorted(dist_data.keys())

    num_plots = len(pts_sorted)
    cols = min(3, num_plots)
    rows = int(np.ceil(num_plots / cols))

    fig_width = max(5 * cols, 8)
    fig_height = 5 * rows
    fig, axes = plt.subplots(
        rows, cols, figsize=(fig_width, fig_height), constrained_layout=True
    )

    fig.suptitle(f"Density Plots {title_suffix}", fontsize=16)

    if num_plots == 1:
        axes = np.array([axes])

    cat_labels = ["Unstim", "Stim True", "Stim Pred"]
    cat_colors = ["blue", "red", "green"]

    for i, (pt, ax) in enumerate(zip(pts_sorted, axes.flatten())):
        for label, color in zip(cat_labels, cat_colors):
            arr = dist_data[pt][label]
            if arr.size > 0:
                sns.kdeplot(
                    arr,
                    ax=ax,
                    label=f"{label} (n={arr.size})",
                    color=color,
                    fill=False,  # set tot True to fill the area under the curve
                    alpha=0.3,
                )

        ax.set_title(f"Patient: {pt}", fontsize=14)
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)
    for j in range(i + 1, len(axes.flatten())):
        fig.delaxes(axes.flatten()[j])

    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def plot_result(prediction_path, original_path, marker, outdir_path,doms_stim):
    target = ad.read(prediction_path)
    target1 = target[:, marker].copy()
    stim = pd.Series(
        target1[(target1.obs["drug"] == doms_stim) & (target1.obs["state"] == 'true_corrected')].X.flatten(), name="Stim True"
    )
    unstim = pd.Series(
        target1[target1.obs["drug"] == "Unstim"].X.flatten(), name="Unstim"
    )
    #dataf = pd.read_csv(original_path)
    pred=pd.Series(
        target1[(target1.obs["drug"] == doms_stim) & (target1.obs["state"] == 'predicted')].X.flatten(), name="Stim Pred"
    )
    print('loaded')
    dist_data = {
        "Patient_1": {
            "Stim True": stim.values,
            "Stim Pred": pred,
            "Unstim": unstim.values,
        }
    }

    create_density_plots(dist_data, outdir_path, title_suffix="")
    return

marker_list=['149Sm_pCREB', '167Er_pERK12', '164Dy_IkB', '159Tb_pMAPKAPK2', '166Er_pNFkB', '151Eu_pp38','155Gd_pS6', '153Eu_pSTAT1', '154Sm_pSTAT3', '150Nd_pSTAT5', '168Yb_pSTAT6', '174Yb_HLADR', '169Tm_CD25']
perio_stim_list_=['TNFa','P._gingivalis']
perio_cell_list_=['Granulocytes_(CD45-CD66+)','B-Cells_(CD19+CD3-)','Classical_Monocytes_(CD14+CD16-)','MDSCs_(lin-CD11b-CD14+HLADRlo)','mDCs_(CD11c+HLADR+)','pDCs(CD123+HLADR+)','Intermediate_Monocytes_(CD14+CD16+)','Non-classical_Monocytes_(CD14-CD16+)','CD56+CD16-_NK_Cells','CD56loCD16+NK_Cells','NK_Cells_(CD7+)','CD4_T-Cells','Tregs_(CD25+FoxP3+)','CD8_T-Cells','CD8-CD4-_T-Cells']
for cell_type in perio_cell_list_:
    for stim in perio_stim_list_:
        if [stim, cell_type] not in [['P._gingivalis', 'Non-classical_Monocytes_(CD14-CD16+)'],['P._gingivalis', 'NK_Cells_(CD7+)'],['TNFa', 'Granulocytes_(CD45-CD66+)']]:
            for marker in marker_list:
                if stim=='P._gingivalis':
                    doms_stim='LPS'
                else:
                    doms_stim=stim
            
                prediction_path = f"results/perio_surge_training_corrected/{stim}_{cell_type}/model-cellot/pred_surge_corrected.h5ad"
                original_path=f"./datasets/surge_concatenated/surge_concatenated_{doms_stim}_{cell_type}.h5ad"
                output_path = f"results/perio_surge_training_corrected/{stim}_{cell_type}/model-cellot/surge_plot/{doms_stim}_{cell_type}_{marker}_surge.png"
                
                plot_result(prediction_path,original_path,marker,output_path,doms_stim)
                print(f"Plot {marker} for {cell_type} and {doms_stim}")