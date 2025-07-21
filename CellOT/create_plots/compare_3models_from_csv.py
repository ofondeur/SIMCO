import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

model = 'original1'
MODEL_BASE_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_{model}"
outdir_path=f"{MODEL_BASE_DIR}/plots3/mae_mmd.csv"
results_df=pd.read_csv(outdir_path)

palette = {
    "CellOT": "#c1443c",   
    "Identity": "#4e4e50",   
    "ReturnTrain": "#76c7c0"  
}


for metric in ['MAE', 'MMD']:
    metric_df = results_df[results_df['metric'] == metric]

    #plt.figure(figsize=(1.2, 1.6))
    fig, ax = plt.subplots(figsize=(1.5, 1.9))
    sns.barplot(
        data=metric_df,
        x='model',
        y='value',
        palette=palette,
        errorbar='se',
        capsize=0.1,
        ax=ax
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([""] * len(metric_df["model"].unique()))
    ax.tick_params(axis='y', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    
    ax.legend().remove()
    ax.set_title("")
    ax.set_ylim(bottom=0)

    # Ajuste la taille de la zone dessinée pour qu’elle remplisse quasiment toute la figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Sauvegarde
    out_path = f"{MODEL_BASE_DIR}/plots3/bar_model_comparison_{metric}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
