#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MaxNLocator
model='original_20marks'
BASE_TRANSFORMED = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/plots_cv_stim_models/cross_validation_{model}"

# Output Directory for Plots
PLOT_OUTPUT_DIR = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/plots_cv_stim_models/cross_validation_{model}/plots_corr"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

valid_jobs_paths="/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/valid_jobs.txt"


# ===== Helper Function =====

def plot_correlation(df_feature, title, save_path):
    if df_feature.empty or df_feature['median_meas'].isna().all() or df_feature['median_pred'].isna().all():
        print(f"  Skipping plot for '{title}': Not enough valid data points.")
        return

    # Drop NaNs pairwise for correlation calculation
    valid_data = df_feature[['median_meas', 'median_pred']].dropna()
    if len(valid_data) < 2:
        print(f"  Skipping plot for '{title}': Fewer than 2 valid paired data points for correlation.")
        return

    spear_corr, spear_p_value = spearmanr(valid_data['median_meas'], valid_data['median_pred'])
    # Calculate Pearson correlation
    pear_corr, pear_p_value = pearsonr(valid_data['median_meas'], valid_data['median_pred'])
    
    plt.figure(figsize=(5, 5))
    ax = sns.scatterplot(data=df_feature, x='median_meas', y='median_pred', s=30, alpha=0.9,edgecolor="#1a1a1a",color="#2f2f2f")
    # Add identity line (y=x)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # Ensure limits are sensible if data is constant
    if lims[0] == lims[1]:
        lims = [lims[0] - 0.1, lims[1] + 0.1] # Add some padding if min==max
    ax.plot(lims, lims, alpha=0.7, zorder=0, linewidth=1, linestyle='--', c='#d3d3d3')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    locator = MaxNLocator(nbins=4, prune='both')  # Try to keep exactly 4 ticks
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    # Set labels and title
    ax.set_xlabel("Ground Truth Median (Stim - Unstim)")
    ax.set_ylabel("Predicted Median (Stim - Unstim)")
    
    plot_title = (f"{title}\n"
                  f"Spearman ρ = {spear_corr:.3f}" + (f" (p={spear_p_value:.2g})" if spear_p_value < 0.05 else "") + "\n"
                  f"Pearson r = {pear_corr:.3f}" + (f" (p={pear_p_value:.2g})" if pear_p_value < 0.05 else "") )
    lin = LinearRegression().fit(X=np.reshape(df_feature['median_meas'], (-1, 1)), y=df_feature['median_pred'])
    ax.plot(lims, lin.predict([[lims[0]], [lims[1]]]), c='#b4313e', alpha=.7)
    ax.set_title(plot_title, fontsize=10)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='both', length=4, width=1, labelsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {os.path.basename(save_path)}")
    plt.close()

couples=[['CD4Teff', 'pSTAT6'], ['CD4Teff', 'pSTAT3'], ['CD4negCD8negTcells', 'pSTAT6'], ['CD8Teff', 'pSTAT6'], ['CD8Tcells Th1', 'pSTAT6'], ['Bcells', 'pSTAT6'], ['mDCs', 'pSTAT6'], ['CD56hiCD16negNK', 'pSTAT6'], ['NKT', 'pSTAT6'], ['CD4Tem', 'pSTAT6'], ['CD8Tem', 'pSTAT6'], ['CD4Tem CCR2+', 'pSTAT6'], ['CD4Tcm CCR2+', 'pSTAT5'], ['CD8Tcm CCR2+', 'pSTAT5'], ['CD4Tcm', 'pSTAT5'], ['CD8Tcm CCR2+', 'pSTAT6'], ['CD4Tem CCR2+', 'pSTAT5'], ['CD4Tem', 'pSTAT5'], ['CD8Tem CCR2+', 'pSTAT6'], ['CD4Tregs', 'pSTAT5']]
# ===== Main Execution =====
def main():
    for couple in couples:
        population,marker=couple[0],couple[1]
        stim='IL246'
        MEAS_PATH = os.path.join(BASE_TRANSFORMED, f"ptb-drugscreen_{stim}_transformed.csv")
        PRED_PATH = os.path.join(BASE_TRANSFORMED, f"ptb-drugscreen_{stim}_predicted_transformed.csv")
        # --- 1. Load and Merge Data ---
        print("Loading data...")
        try:
            m = pd.read_csv(MEAS_PATH)
            p = pd.read_csv(PRED_PATH)
            print(f"Loaded measured data ({m.shape}), predicted data ({p.shape})")
        except FileNotFoundError as e:
            sys.exit(f"Error loading files: {e}")
    
        # Ensure required columns exist - assuming 'sampleID','population','marker','stim','median'
        required_cols = ['sampleID', 'population', 'marker', 'stim', 'median']
        if not all(col in m.columns for col in required_cols):
             sys.exit(f"Error: Measured data missing required columns ({required_cols}). Found: {m.columns.tolist()}")
        if not all(col in p.columns for col in required_cols):
             sys.exit(f"Error: Predicted data missing required columns ({required_cols}). Found: {p.columns.tolist()}")
    
        # Filter out Unstim (we are comparing deltas)
        m = m[m['stim'] != "Unstim"].copy()
        p = p[p['stim'] != "Unstim"].copy()
    
        merged_df = pd.merge(m, p,
                             on=["sampleID", "population", "marker", "stim"],
                             suffixes=("_meas", "_pred"),
                             how='inner') # Use inner merge to only keep features present in both
    
        print(f"Merged data shape (inner join): {merged_df.shape}")
        if merged_df.empty:
            sys.exit("Error: Merged DataFrame is empty. Check if keys match between input files.")
    
        print("\nGenerating Spearman correlation plots for highlighted features...")
        num_plots_generated = 0

        # Filter the merged dataframe for this specific feature
        df_feature = merged_df[
            (merged_df['stim'] == stim) &
            (merged_df['population'] == population) &
            (merged_df['marker'] == marker)
        ].copy()

        if df_feature.empty:
            print(f"  Skipping plot for {stim}/{population}/{marker}: No data after merge.")
            continue

        plot_title = f"{stim} | {population} | {marker}"
        safe_filename = f"{stim}_{population}_{marker}".replace(" ", "_").replace("/", "-").replace("+", "pos") + ".pdf"
        save_path = os.path.join(PLOT_OUTPUT_DIR, safe_filename)

        # Create the plot
        plot_correlation(df_feature, plot_title, save_path)
        num_plots_generated += 1

    print(f"\nGenerated {num_plots_generated} Spearman correlation plots in: {PLOT_OUTPUT_DIR}")

if __name__ == "__main__":
    main()