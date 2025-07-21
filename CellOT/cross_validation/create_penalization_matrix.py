#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ===== Configuration =====
# Input CSV path (Ground Truth Deltas)
MEAS_PATH = "/Users/peter/Documents/LocalScripts/ot/Drugscreen/ptb/lopo10_transformed/ptb_transformed.csv"

# Output Directory
OUTPUT_DIR = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/cross_validation/feats_selection" # New output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output filenames
KNOWLEDGE_TABLE_CSV = os.path.join(OUTPUT_DIR, "marker_inclusion_knowledge_table.csv")
PLOT_SUBDIR = os.path.join(OUTPUT_DIR, "stim_median_delta_tables")
os.makedirs(PLOT_SUBDIR, exist_ok=True)

# Threshold for highlighting small changes and for inclusion knowledge table
DELTA_THRESHOLD = 0.1
IL33_THRESHOLD = 0.01

# ===== Constants =====
# Assuming the CSV contains functional markers relevant to PTB
# If not, load features.txt or define explicitly
# Let's define based on common markers likely in the file
FUNCTIONAL_MARKERS = sorted(['pCREB', 'pSTAT5', 'pP38', 'PD1', 'pSTAT1', 'pSTAT3', 'pS6',
       'CD44', 'CD36', 'PDL1', 'pMK2', 'GLUT1', 'IkB', 'pNFkB', 'pERK',
       'pSTAT6', 'CD25', 'pPLCg', 'pSTAT4', 'HLADR']) # Use the full list expected in ptb_transformed.csv

CELL_TYPES = sorted(list({ # Use the PTB list
    "Granulocytes","Bcells","cMCs","MDSCs","mDCs","pDCs","intMCs","ncMCs",
    "CD56hiCD16negNK","CD56loCD16posNK","NK cells CD11c-","NK cells CD11c+",
    "CD4Tnaive","CD4Teff","CD4Tcm","CD4Tcm CCR2+","CD4Tem","CD4Tem CCR2+",
    "CD4Tregs","CD8Tcm","CD8Tcm CCR2+","CD8Tem","CD8Tem CCR2+",
    "CD8Tnaive","CD8Teff","CD8Tcells Th1","CD4negCD8negTcells","NKT"
}))

STIMS = sorted(["TNFa", "LPS", "IL246", "IFNa", "GMCSF", "PI", "IL33"]) # Stims to analyze

# ===== Helper Function for Table Plot =====

def plot_feature_table(data_pivot, title, save_path, threshold=DELTA_THRESHOLD):
    """
    Creates a table plot from a pivoted DataFrame, highlighting small absolute values.
    """
    print(f"Generating table plot: {title}")
    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    fig, ax = plt.subplots(figsize=(max(10, len(data_pivot.columns)*0.6), # Adjust figsize based on columns
                                  max(7.5, len(data_pivot.index)*0.1))) # Adjust figsize based on rows
    ax.set_axis_off() # Hide the axes

    # Use plt.table - more control than heatmap text
    table = plt.table(cellText=data_pivot.round(2).astype(str).values, # Format values
                        rowLabels=data_pivot.index,
                        colLabels=data_pivot.columns,
                        loc='center',
                        cellLoc='center',
                        rowLoc='right')

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2) # Adjust scale if needed

    # Iterate through cells to set color based on threshold
    for (i, j), cell in table.get_celld().items():
        # Skip header cells
        if i == 0 or j == -1:
            cell.set_text_props(weight='bold')
            cell.set_facecolor("#E0E0E0") # Light grey header
            continue

        try:
            # Get numeric value corresponding to the cell
            # i-1 for row index, j for column index in the DataFrame
            numeric_value = data_pivot.iloc[i-1, j]
            if pd.isna(numeric_value):
                 cell.set_text_props(color='grey') # Grey out NaN text
                 cell.set_facecolor('#F5F5F5') # Very light grey background for NaN
            # Highlight if absolute value is below threshold
            elif abs(numeric_value) < threshold:
                #cell.set_text_props(color='red', weight='bold') # Highlight text in red
                cell.set_facecolor('#FFCCCC') # Light red background
            else:
                 cell.set_text_props(color='black') # Default text color
        except (IndexError, ValueError):
            # Handle cases where getting the value fails
             cell.set_text_props(color='grey')


    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.tight_layout() # Adjust layout
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved table plot to {os.path.basename(save_path)}")
    plt.close(fig)


# ===== Main Execution =====
def main():
    # --- 1. Load and Prepare Data ---
    print(f"Loading ground truth median deltas from: {MEAS_PATH}")
    try:
        df_meas = pd.read_csv(MEAS_PATH)
        # Ensure correct column names
        expected_cols = ['sampleID', 'population', 'marker', 'stim', 'median']
        if not all(col in df_meas.columns for col in expected_cols):
             raise ValueError(f"CSV must contain columns: {expected_cols}")
        print(f"Loaded measured data with shape: {df_meas.shape}")
    except FileNotFoundError:
        sys.exit(f"Error: Input CSV not found at {MEAS_PATH}")
    except ValueError as ve:
         sys.exit(f"Error: {ve}")
    except Exception as e:
        sys.exit(f"Error loading CSV: {e}")

    # Filter out Unstim if present (shouldn't be if it's delta, but safety check)
    df_deltas = df_meas[df_meas['stim'] != 'Unstim'].copy()
    if df_deltas.empty:
        sys.exit("Error: No non-Unstim data found in the input file.")

    # Filter for expected markers and populations (optional, defensive)
    df_deltas = df_deltas[df_deltas['marker'].isin(FUNCTIONAL_MARKERS)]
    df_deltas = df_deltas[df_deltas['population'].isin(CELL_TYPES)]
    print(f"Filtered data shape (functional markers, expected cell types): {df_deltas.shape}")


    # --- 2. Calculate Median Delta Across Samples ---
    print("\nCalculating median delta across samples for each feature...")
    # Group by stim, population, marker and find the median of the 'median' (delta) column
    median_deltas = df_deltas.groupby(['stim', 'population', 'marker'], observed=True)['median'].median().reset_index()
    median_deltas = median_deltas.rename(columns={'median': 'median_delta'}) # Rename for clarity
    print(f"Calculated {median_deltas.shape[0]} median delta values.")

    # --- 3. Generate Knowledge Table CSV ---
    print("\nGenerating marker inclusion knowledge table...")
    knowledge_df = median_deltas.copy()
    # Add boolean flag: True if abs(median_delta) >= threshold
    knowledge_df['include_marker'] = (
        (np.abs(knowledge_df['median_delta']) >= DELTA_THRESHOLD) |
        ((np.abs(knowledge_df['median_delta']) >= IL33_THRESHOLD) & (knowledge_df['stim'] == "IL33"))
    )
    # Keep only relevant columns
    knowledge_df = knowledge_df[['stim', 'population', 'marker', 'include_marker', 'median_delta']] # Added median delta for context

    try:
        knowledge_df.to_csv(KNOWLEDGE_TABLE_CSV, index=False)
        print(f"Saved knowledge table to: {KNOWLEDGE_TABLE_CSV}")
    except Exception as e:
        print(f"Error saving knowledge table CSV: {e}")

    # --- 4. Generate Table Plots per Stim ---
    print("\nGenerating table plots per stimulation...")
    # Ensure columns/rows are ordered consistently
    pivot_rows = CELL_TYPES
    pivot_cols = FUNCTIONAL_MARKERS

    for stim in STIMS:
        stim_data = median_deltas[median_deltas['stim'] == stim]
        if stim_data.empty:
            print(f"No data found for stimulation '{stim}'. Skipping plot.")
            continue

        # Pivot the data for the table plot
        try:
            data_pivot = stim_data.pivot(index='population', columns='marker', values='median_delta')
            # Reindex to ensure all expected rows/columns are present and ordered
            data_pivot = data_pivot.reindex(index=pivot_rows, columns=pivot_cols)
        except Exception as e:
            print(f"Error pivoting data for stim '{stim}': {e}. Skipping plot.")
            continue

        if stim == "IL33":
            threshold = IL33_THRESHOLD
        else:
            threshold = DELTA_THRESHOLD

        # Define plot title and save path
        plot_title = f"Median Delta ({stim} - Unstim) Across Samples. (Abs < {threshold} highlighted red)"
        safe_stim_name = stim.replace(" ", "_").replace("/", "-")
        save_path = os.path.join(PLOT_SUBDIR, f"median_delta_table_{safe_stim_name}.png")

        # Create the plot
        plot_feature_table(data_pivot, plot_title, save_path, threshold=threshold)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()