#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

# ===== Configuration =====
# Input CSV paths (Adjust if your filenames changed)
BASE_TRANSFORMED = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/cellwise/cross_validation_olivier"

# Output Directory for Plots
PLOT_OUTPUT_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/cellwise/cross_validation_olivier/plots"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

valid_jobs_paths="/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/valid_jobs.txt"


sns.set(style="whitegrid")
plt.rcParams.update({
    'font.family': 'Helvetica', # Or 'Arial', 'sans-serif'
    'font.size': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'axes.linewidth': 1,
    'axes.edgecolor':'gray'
})


# ===== Main Execution =====
def main():
    for stim,cell, sanitized_cell in read(valid_jobs_paths):
            with open(f'/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/features_evaluation/features_{stim}_{sanitized_cell}.txt') as f:
                    markers = [line.strip() for line in f if line.strip()]
            for marker in markers:
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
            
    
                required_cols = ['sampleID', 'population', 'marker', 'stim', 'median']
                if not all(col in m.columns for col in required_cols):
                     sys.exit(f"Error: Measured data missing required columns ({required_cols}). Found: {m.columns.tolist()}")
                if not all(col in p.columns for col in required_cols):
                     sys.exit(f"Error: Predicted data missing required columns ({required_cols}). Found: {p.columns.tolist()}")
            
                # Filter out Unstim (we are comparing deltas)
                m = m[m['stim'] != "Unstim"].copy()
                p = p[p['stim'] != "Unstim"].copy()
            
                # Merge ground truth and predictions
                merged_df = pd.merge(m, p,
                                     on=["sampleID", "population", "marker", "stim"],
                                     suffixes=("_meas", "_pred"),
                                     how='inner') # Use inner merge to only keep features present in both
            
                print(f"Merged data shape (inner join): {merged_df.shape}")
                if merged_df.empty:
                    sys.exit("Error: Merged DataFrame is empty. Check if keys match between input files.")
            
                # --- 2. Generate Plots for Hand-picked Features ---
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
    
                # Define plot title and save path
                plot_title = f"{stim} | {population} | {marker}"
                safe_filename = f"{stim}_{population}_{marker}".replace(" ", "_").replace("/", "-").replace("+", "pos") + ".png"
            
            num_plots_generated += 1

if __name__ == "__main__":
    main()