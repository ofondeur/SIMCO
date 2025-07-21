#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
############################
# Helper Functions
############################
INVERSE_CELL_TYPE_MAPPING = {
    "granulocytes": "Granulocytes",
    "granulocytes (cd45-cd66+)": "Granulocytes",
    "neutrophils": "Granulocytes",
    "bcells": "Bcells",
    "b-cells (cd19+cd3-)": "Bcells",
    "b-cells": "Bcells",
    "cmc": "cMCs",
    "classical monocytes (cd14+cd16-)": "cMCs",
    "cmcs": "cMCs",
    "mdsc": "MDSCs",
    "mdscs (lin-cd11b-cd14+hladrlo)": "MDSCs",
    "mdscs": "MDSCs",
    "mdc": "mDCs",
    "mdcs (cd11c+hladr+)": "mDCs",
    "mdcs": "mDCs",
    "pdc": "pDCs",
    "pdcs(cd123+hladr+)": "pDCs",
    "pdcs": "pDCs",
    "intmc": "intMCs",
    "intermediate monocytes (cd14+cd16+)": "intMCs",
    "intmcs": "intMCs",
    "cd56brightcd16negnk": "CD56hiCD16negNK",
    "cd56+cd16- nk cells": "CD56hiCD16negNK",
    "cd56hicd16negnk": "CD56hiCD16negNK",
    "cd16- cd56+ nk": "CD56hiCD16negNK",
    "cd56dimcd16posnk": "CD56loCD16posNK",
    "cd56locd16+nk cells": "CD56loCD16posNK",
    "cd16+ cd56lo nk": "CD56loCD16posNK",
    "cd4tcells": "CD4Tcells",
    "cd4 t-cells": "CD4Tcells",
    "cd4 tcells": "CD4Tcells",
    "tregs": "Tregs",
    "treg": "Tregs",
    "tregs (cd25+foxp3+)": "Tregs",
    "cd8tcells": "CD8Tcells",
    "cd8 t-cells": "CD8Tcells",
    "cd8 tcells": "CD8Tcells",
    "cd4negcd8neg": "CD4negCD8negTcells",
    "cd8-cd4- t-cells": "CD4negCD8negTcells",
    "cd4- cd8- t-cells": "CD4negCD8negTcells",
    "cd4negcd8negtcells": "CD4negCD8negTcells"
}

CELL_TYPE_MAPPING = {
    "granulocytes": "Granulocytes",
    "granulocytes (cd45-cd66+)": "Granulocytes",
    "neutrophils": "Granulocytes",
    "bcells": "Bcells",
    "b-cells (cd19+cd3-)": "Bcells",
    "b-cells": "Bcells",
    "ncmc": "ncMCs",
    "non-classical monocytes (cd14-cd16+)": "ncMCs",
    "ncmcs": "ncMCs",
    "cmc": "cMCs",
    "classical monocytes (cd14+cd16-)": "cMCs",
    "cmcs": "cMCs",
    "mdsc": "MDSCs",
    "mdscs (lin-cd11b-cd14+hladrlo)": "MDSCs",
    "mdscs": "MDSCs",
    "mdc": "mDCs",
    "mdcs (cd11c+hladr+)": "mDCs",
    "mdcs": "mDCs",
    "pdc": "pDCs",
    "pdcs(cd123+hladr+)": "pDCs",
    "pdcs": "pDCs",
    "intmc": "intMCs",
    "intermediate monocytes (cd14+cd16+)": "intMCs",
    "intmcs": "intMCs",
    "cd56brightcd16negnk": "CD56hiCD16negNK",
    "cd56+cd16- nk cells": "CD56hiCD16negNK",
    "cd56hicd16negnk": "CD56hiCD16negNK",
    "cd16- cd56+ nk": "CD56hiCD16negNK",
    "cd56dimcd16posnk": "CD56loCD16posNK",
    "cd56locd16+nk cells": "CD56loCD16posNK",
    "cd16+ cd56lo nk": "CD56loCD16posNK",
    "nk": "NKcells",
    "nk cells (cd7+)": "NKcells",
    "cd4tcells": "CD4Tcells",
    "cd4 t-cells": "CD4Tcells",
    "cd4 tcells": "CD4Tcells",
    "tregs": "Tregs",
    "treg": "Tregs",
    "tregs (cd25+foxp3+)": "Tregs",
    "cd8tcells": "CD8Tcells",
    "cd8 t-cells": "CD8Tcells",
    "cd8 tcells": "CD8Tcells",
    "cd4negcd8neg": "CD4negCD8negTcells",
    "cd8-cd4- t-cells": "CD4negCD8negTcells",
    "cd4- cd8- t-cells": "CD4negCD8negTcells",
    "cd4negcd8negtcells": "CD4negCD8negTcells"
}
def normalize_cell_type(cell):
    key = cell.strip().lower()
    if key not in INVERSE_CELL_TYPE_MAPPING:
        raise AssertionError(f"Unrecognized cell type: {cell}")
    return INVERSE_CELL_TYPE_MAPPING[key]

STIM_MAPPING = {
    "unstim": "Unstim",
    "tnfa": "TNFa",
    "lps": "LPS",
    "p. gingivalis": "LPS",
    "ifna": "IFNa"
}

def normalize_stim(stim):
    key = stim.strip().lower()
    if key not in STIM_MAPPING:
        raise AssertionError(f"Unrecognized stimulation: {stim}")
    return STIM_MAPPING[key]

def normalize_marker(raw_marker):
    marker = raw_marker.strip().lower()
    # Remove leading "p" characters (except for "pp38")
    while marker.startswith("p"):
        marker = marker[1:]
    if marker == "creb":
        return "pCREB"
    elif marker in ["erk", "erk12"]:
        return "pERK"
    elif marker == "ikb":
        return "IkB"
    elif marker in ["mk2", "mapkapk2"]:
        return "pMK2"
    elif marker == "nfkb":
        return "pNFkB"
    elif marker in ["38", "p38"]:
        return "pp38"
    elif marker == "s6":
        return "pS6"
    elif marker == "stat1":
        return "pSTAT1"
    elif marker == "stat3":
        return "pSTAT3"
    elif marker == "stat5":
        return "pSTAT5"
    elif marker == "stat6":
        return "pSTAT6"
    elif marker == "hladr":
        return "HLADR"
    elif marker == "cd25":
        return "CD25"
    else:
        raise AssertionError(f"Unrecognized marker: {raw_marker}")
        
def load_training_means(perio_path):
    """
    Load the training (perio) data and compute the mean for each (stim, marker, population)
    combination. Only non-Unstim rows are used.
    """
    df = pd.read_csv(perio_path)
    df = df[df["stim"] != "Unstim"]
    means = df.groupby(["stim", "marker", "population"])["median"].mean().reset_index()
    mean_dict = { (normalize_stim(row["stim"]), normalize_marker(row["marker"]), row["population"]): row["median"]
                  for _, row in means.iterrows() }
    return mean_dict

def compute_naive_errors(test_df, mean_dict, is_surge=False):
    """
    For each non-Unstim row in the test data, lookup the training mean (naive prediction)
    and compute absolute error. For surge we only compare BL samples.
    """
    df = test_df[test_df["stim"] != "Unstim"].copy()
    if is_surge:
        df = df[df["sampleID"].str.endswith("BL")]
    def lookup(row):
        key = (row["stim"], row["marker"], row["population"])
        if key not in mean_dict:
            raise AssertionError(f"Training mean not found for combination: {key}")
        return mean_dict[key]
    df["pred_mean"] = df.apply(lookup, axis=1)
    df["error"] = np.abs(df["median"] - df["pred_mean"])
    df["method"] = "Mean"
    return df[["sampleID", "population", "marker", "stim", "error", "method"]]

def compute_pair_errors_surge(test_df):
    """
    For surge, compute the error between paired measurements (BL vs IDX).
    """
    df = test_df[test_df["stim"] != "Unstim"].copy()
    df["patient"] = df["sampleID"].apply(lambda x: x.split("_")[0])
    df["timepoint"] = df["sampleID"].apply(lambda x: x.split("_")[1])
    pivot = df.pivot_table(index=["patient", "stim", "marker", "population"],
                           columns="timepoint", values="median", aggfunc="first").reset_index()
    pivot = pivot.dropna(subset=["BL", "IDX"])
    pivot["error"] = np.abs(pivot["BL"] - pivot["IDX"])
    pivot["method"] = "Pair"
    pivot["sampleID"] = pivot["patient"]
    return pivot[["sampleID", "population", "marker", "stim", "error", "method"]]

def compute_ot_errors(measured_path, predicted_path, is_surge=False):
    """
    Compute OT errors by merging measured and predicted CSVs on (sampleID, population, marker, stim)
    and taking the absolute difference of the median values.
    For surge, only BL samples are considered.
    """
    measured_df = pd.read_csv(measured_path)
    predicted_df = pd.read_csv(predicted_path)
    measured_df = measured_df[measured_df["stim"] != "Unstim"].copy()
    predicted_df = predicted_df[predicted_df["stim"] != "Unstim"].copy()
    if is_surge:
        measured_df = measured_df[measured_df["sampleID"].str.endswith("BL")]
        predicted_df = predicted_df[predicted_df["sampleID"].str.endswith("BL")]
    merged = pd.merge(measured_df, predicted_df, on=["sampleID", "population", "marker", "stim"],
                      suffixes=("_meas", "_pred"))
    merged["error"] = np.abs(merged["median_meas"] - merged["median_pred"])
    merged["method"] = "OT"
    return merged[["sampleID", "population", "marker", "stim", "error", "method"]]

def compute_ot_errors_idx(measured_path, predicted_path):
    """
    For surge, compute OT errors using IDX samples instead of BL.
    """
    measured_df = pd.read_csv(measured_path)
    predicted_df = pd.read_csv(predicted_path)
    measured_df = measured_df[measured_df["stim"] != "Unstim"].copy()
    predicted_df = predicted_df[predicted_df["stim"] != "Unstim"].copy()
    measured_df = measured_df[measured_df["sampleID"].str.endswith("IDX")]
    predicted_df = predicted_df[predicted_df["sampleID"].str.endswith("IDX")]
    merged = pd.merge(measured_df, predicted_df, on=["sampleID", "population", "marker", "stim"],
                      suffixes=("_meas", "_pred"))
    merged["error"] = np.abs(merged["median_meas"] - merged["median_pred"])
    merged["method"] = "OT_IDX"
    return merged[["sampleID", "population", "marker", "stim", "error", "method"]]

def plot_error_distribution(data, title, save_path, methods_order):
    plt.rcParams.update({
        'font.family': 'Helvetica',
        'font.size': 14,
        'axes.edgecolor': 'gray'
    })
    plt.figure(figsize=(6, 4))
    data_by_method = [data[data["method"] == m]["error"] for m in methods_order]
    bp = plt.boxplot(data_by_method, positions=range(len(methods_order)), widths=0.6, patch_artist=True,
                     showfliers=False)
    for box in bp['boxes']:
        box.set(facecolor='none', edgecolor='black', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)
    for i, m in enumerate(methods_order):
        y = data[data["method"]==m]["error"].values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.7, edgecolor='gray', facecolor='white', s=20)
    plt.xticks(range(len(methods_order)), methods_order)
    plt.ylabel("Absolute Error")
    plt.title(title)
    # plt.ylim(0, 1)
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_plots(error_df, output_folder, methods_order, study):
    subfolders = ["individual", "stims", "markers", "populations", "combined"]
    for sub in subfolders:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)
    combined_path = os.path.join(output_folder, "combined", "combined_error_plot.png")
    plot_error_distribution(error_df, f"{study} Combined", combined_path, methods_order)
    stims = sorted(error_df["stim"].unique())
    populations_to_plot = ["mDCs"]
    markers_to_plot = ["pCREB","pp38","pS6","pERK","pNFkB","pSTAT3"]
    stims_to_plot = sorted(error_df["stim"].unique())
    for stim in stims_to_plot:
        for marker in markers_to_plot:
            for pop in populations_to_plot:
                subset = error_df[(error_df["stim"] == stim) &
                                    (error_df["marker"] == marker) &
                                    (error_df["population"] == pop)]
                if subset.empty:
                    continue
                title = f"{study} | {stim} | {marker} | {pop}"
                filename = f"{stim}_{marker}_{pop}.png".replace(" ", "_")
                save_path = os.path.join(output_folder, "individual", filename)
                plot_error_distribution(subset, title, save_path, methods_order)
    for stim in stims:
        subset = error_df[error_df["stim"] == stim]
        if subset.empty:
            continue
        title = f"{study} | Stim: {stim}"
        filename = f"{stim}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "stims", filename)
        plot_error_distribution(subset, title, save_path, methods_order)
    markers = sorted(error_df["marker"].unique())
    for marker in markers:
        subset = error_df[error_df["marker"] == marker]
        if subset.empty:
            continue
        title = f"{study} | Marker: {marker}"
        filename = f"{marker}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "markers", filename)
        plot_error_distribution(subset, title, save_path, methods_order)
    populations = sorted(error_df["population"].unique())
    for pop in populations:
        subset = error_df[error_df["population"] == pop]
        if subset.empty:
            continue
        title = f"{study} | Population: {pop}"
        filename = f"{pop}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "populations", filename)
        plot_error_distribution(subset, title, save_path, methods_order)

def apply_penalization_filter(error_df, penalization_folder, study):
    penalizations = {}
    for stim in error_df["stim"].unique():
        stim_lower = stim.lower()
        file_path = os.path.join(penalization_folder, f"{stim_lower}.csv")
        if os.path.exists(file_path):
            df_pen = pd.read_csv(file_path, index_col=0)
            penalizations[stim] = df_pen
        else:
            raise AssertionError(f"Penalization file for stim {stim} not found at {file_path}")
    def get_pen(row):
        stim = row["stim"]
        population = row["population"]
        marker = row["marker"]
        pen_df = penalizations[stim]
        if population not in pen_df.index:
            raise AssertionError(f"Population {population} not found in penalization file for {stim}")
        if marker not in pen_df.columns:
            raise AssertionError(f"Marker {marker} not found in penalization file for {stim}")
        return pen_df.loc[population, marker]
    error_df["pen"] = error_df.apply(get_pen, axis=1)
    mean_error_df = error_df[error_df["method"] == "Mean"]
    total = len(mean_error_df)
    zero_before = (mean_error_df["error"] == 0).sum()
    nonzero_before = total - zero_before
    filtered_df = error_df[error_df["pen"] == 1].copy()
    removed_df = error_df[error_df["pen"] == 0].copy()
    mean_filtered_df = filtered_df[filtered_df["method"] == "Mean"]
    mean_removed_df = removed_df[removed_df["method"] == "Mean"]
    total_after = len(mean_filtered_df)
    zero_after = (mean_filtered_df["error"] == 0).sum()
    nonzero_after = total_after - zero_after
    removed_total = len(mean_removed_df)
    removed_zero = (mean_removed_df["error"] == 0).sum()
    removed_nonzero = removed_total - removed_zero
    print(f"----- {study} Penalization Summary -----")
    print(f"Before filtering: Total = {total}, Zero error = {zero_before}, Non-zero error = {nonzero_before}")
    print(f"Removed by penalization: Total = {removed_total}, Zero error = {removed_zero}, Non-zero error = {removed_nonzero}")
    print(f"After filtering: Total = {total_after}, Zero error = {zero_after}, Non-zero error = {nonzero_after}")
    print("--------------------------------------------------")
    return filtered_df

def compute_summary_metrics(error_df):
    summary = error_df.groupby(["stim", "marker", "population", "method"]).agg(
                MAE=("error", "mean"),
                RMSE=("error", lambda x: np.sqrt(np.mean(x**2)))
             ).reset_index()
    return summary

def plot_summary_metric_distribution(data, metric, title, save_path, methods_order):
    plt.rcParams.update({
        'font.family': 'Helvetica',
        'font.size': 14,
        'axes.edgecolor': 'gray'
    })
    plt.figure(figsize=(6, 4))
    data_by_method = [data[data["method"] == m][metric] for m in methods_order]
    bp = plt.boxplot(data_by_method, positions=range(len(methods_order)), widths=0.6, patch_artist=True,
                     showfliers=False)
    for box in bp['boxes']:
        box.set(facecolor='none', edgecolor='black', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)
    for i, m in enumerate(methods_order):
        y = data[data["method"]==m][metric].values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.7, edgecolor='gray', facecolor='white', s=20)
    plt.xticks(range(len(methods_order)), methods_order)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_summary_plots(summary_df, output_folder, metric, methods_order, study):
    subfolders = ["stims", "markers", "populations", "combined"]
    for sub in subfolders:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)
    combined_path = os.path.join(output_folder, "combined", f"combined_{metric}_plot.png")
    plot_summary_metric_distribution(summary_df, metric, f"{study} Combined {metric}", combined_path, methods_order)
    stims = sorted(summary_df["stim"].unique())
    for stim in stims:
        subset = summary_df[summary_df["stim"] == stim]
        if subset.empty:
            continue
        title = f"{study} | Stim: {stim} | {metric}"
        filename = f"{stim}_{metric}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "stims", filename)
        plot_summary_metric_distribution(subset, metric, title, save_path, methods_order)
    markers = sorted(summary_df["marker"].unique())
    for marker in markers:
        subset = summary_df[summary_df["marker"] == marker]
        if subset.empty:
            continue
        title = f"{study} | Marker: {marker} | {metric}"
        filename = f"{marker}_{metric}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "markers", filename)
        plot_summary_metric_distribution(subset, metric, title, save_path, methods_order)
    populations = sorted(summary_df["population"].unique())
    for pop in populations:
        subset = summary_df[summary_df["population"] == pop]
        if subset.empty:
            continue
        title = f"{study} | Population: {pop} | {metric}"
        filename = f"{pop}_{metric}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "populations", filename)
        plot_summary_metric_distribution(subset, metric, title, save_path, methods_order)

############################
# Main Processing Function
############################

def main():
    cohort='surge'
    stim='LPS'
    base_dir = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/cellwise/results_training_sinplecorr/"
    # Use combined_transformed paths
    combined_dir = base_dir
    
    # File paths for training and test data (old measured)
    surge_path = os.path.join(combined_dir, f"surge_{stim}_transformed.csv")
    perio_path = os.path.join(combined_dir, f"perio_{stim}_transformed.csv")
    # File paths for predicted (OT) data
    surge_ot_path = os.path.join(combined_dir, f"surge_{stim}_predicted_transformed.csv")
    t1=pd.read_csv(surge_path)
    t2=pd.read_csv(perio_path)
    #print('surge', list(t1['population'].unique()))
    #print('perio', list(t2['population'].unique()))
    # Load training means from perio
    mean_dict = load_training_means(perio_path)
    
    ############################
    # Process Surge Dataset (BL predictions)
    ############################
    surge_df = pd.read_csv(surge_path)
    surge_mean_errors = compute_naive_errors(surge_df, mean_dict, is_surge=True)
    surge_ot_errors = compute_ot_errors(surge_path, surge_ot_path, is_surge=True)
    surge_pair_errors = compute_pair_errors_surge(surge_df)
    surge_errors = pd.concat([surge_mean_errors, surge_ot_errors, surge_pair_errors], ignore_index=True)
    
    ############################
    # Process Surge Dataset (IDX predictions)
    ############################
    surgeIDX_ot_errors = compute_ot_errors_idx(surge_path, surge_ot_path)
    # For surgeIDX errors we reuse the same mean and pair errors, but we add OT errors for IDX
    surge_errors_idx = pd.concat([surge_mean_errors, surgeIDX_ot_errors, surge_pair_errors], ignore_index=True)
    
    # Save error dataframes (optional)
    surge_errors.to_csv(os.path.join(combined_dir, "surge_errors.csv"), index=False)
    
    ############################
    # Penalization Filtering
    ############################
    penal_dir="/home/groups/gbrice/ptb-drugscreen/ot/cellot_pheno/cells_combined/scripts/plots"
    penalization_folder = os.path.join(base_dir, "penalizations")
    print("Processing Surge penalization filter:")
    surge_errors_filtered = apply_penalization_filter(surge_errors, penalization_folder, "Surge")
    print("Processing SurgeIDX penalization filter:")
    surge_errors_idx_filtered = apply_penalization_filter(surge_errors_idx, penalization_folder, "SurgeIDX")
    
    ############################
    # Plotting Error Distributions
    ############################
    plots_dir = base_dir+ f"{cohort}_{stim}/"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define method order.
    surge_methods_order = ["Mean", "Pair", "OT"]
    surgeIDX_methods_order = ["Mean", "Pair", "OT_IDX"]
    
    # Save plots in subfolders under combined_plots.
    generate_plots(surge_errors_filtered, os.path.join(plots_dir, "surge"), surge_methods_order, "Surge")
    generate_plots(surge_errors_idx_filtered, os.path.join(plots_dir, "surgeIDX"), surgeIDX_methods_order, "SurgeIDX")
    
    ############################
    # Summary Metrics and Plots (MAE and RMSE)
    ############################
    surge_summary = compute_summary_metrics(surge_errors_filtered)
    surgeIDX_summary = compute_summary_metrics(surge_errors_idx_filtered)
    
    # Create output folders for summary plots under combined_plots.
    surge_mae_dir = os.path.join(plots_dir, "surge_MAE_plots")
    surge_rmse_dir = os.path.join(plots_dir, "surge_RMSE_plots")
    surgeIDX_mae_dir = os.path.join(plots_dir, "surgeIDX_MAE_plots")
    surgeIDX_rmse_dir = os.path.join(plots_dir, "surgeIDX_RMSE_plots")
    os.makedirs(surge_mae_dir, exist_ok=True)
    os.makedirs(surge_rmse_dir, exist_ok=True)
    os.makedirs(surgeIDX_mae_dir, exist_ok=True)
    os.makedirs(surgeIDX_rmse_dir, exist_ok=True)
    
    generate_summary_plots(surge_summary, surge_mae_dir, "MAE", surge_methods_order, "Surge MAE")
    generate_summary_plots(surge_summary, surge_rmse_dir, "RMSE", surge_methods_order, "Surge RMSE")
    
    generate_summary_plots(surgeIDX_summary, surgeIDX_mae_dir, "MAE", surgeIDX_methods_order, "SurgeIDX MAE")
    generate_summary_plots(surgeIDX_summary, surgeIDX_rmse_dir, "RMSE", surgeIDX_methods_order, "SurgeIDX RMSE")
    
    print("Summary plots generated.")

if __name__ == "__main__":
    main()