#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_correlations(df, group_cols):
    results = []
    grouped = df.groupby(group_cols + ["method"])
    for group, group_df in grouped:
        if group_df["true"].nunique() <= 1 or group_df["pred"].nunique() <= 1:
            continue
        spearman_corr = spearmanr(group_df["true"], group_df["pred"]).correlation
        pearson_corr = pearsonr(group_df["true"], group_df["pred"])[0]
        row = dict(zip(group_cols, group))
        row.update({
            "method": group_df["method"].iloc[0],
            "spearman": spearman_corr,
            "pearson": pearson_corr
        })
        results.append(row)
    return pd.DataFrame(results)
 
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

def compute_ot_errors(measured_path, predicted_path, method,is_surge=False):
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
    merged["method"] = f"{method}"
    merged["true"] = merged["median_meas"]
    merged["pred"] = merged["median_pred"]
    if 'mmd' not in predicted_df.columns:
        print(f"⚠️ Warning: 'mmd' column not found in {predicted_path}")
        predicted_df['mmd'] = np.nan  # ou raise une erreur si tu préfères

    return merged[["sampleID", "population", "marker", "stim", "error", "true","pred","method","mmd"]]

def plot_correlation_boxplots(corr_df, metric, group_name, methods_order, out_dir, title):
    plt.rcParams.update({
        'font.family': 'Helvetica',
        'font.size': 14,
        'axes.edgecolor': 'gray'
    })
    plt.figure(figsize=(6, 4))
    data_by_method = [corr_df[corr_df["method"] == m][metric] for m in methods_order]
    bp = plt.boxplot(data_by_method, positions=range(len(methods_order)), widths=0.6,
                     patch_artist=True, showfliers=False)
    for box in bp['boxes']:
        box.set(facecolor='none', edgecolor='black', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='black', linewidth=1)
    for i, m in enumerate(methods_order):
        y = corr_df[corr_df["method"]==m][metric].values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.7, edgecolor='gray', facecolor='white', s=20)
    plt.xticks(range(len(methods_order)), methods_order)
    plt.ylabel(f"{metric.capitalize()} Correlation")
    plt.title(title)
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    filename = f"boxplot_{metric}_by_{group_name}.png".replace(" ", "_")
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()


def plot_MMD_distribution(data, title, save_path, methods_order):
    plt.rcParams.update({
        'font.family': 'Helvetica',
        'font.size': 14,
        'axes.edgecolor': 'gray'
    })
    plt.figure(figsize=(6, 4))
    data_by_method = [data[data["method"] == m]["mmd"] for m in methods_order]
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
        y = data[data["method"]==m]["mmd"].values
        x = np.random.normal(i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.7, edgecolor='gray', facecolor='white', s=20)
    plt.xticks(range(len(methods_order)), methods_order)
    plt.ylabel("MMD")
    plt.title(title)
    # plt.ylim(0, 1)
    plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
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

def generate_plots(error_df,output_folder, methods_order, study):
    subfolders = ["individual", "stims", "markers", "populations", "combined"]
    for sub in subfolders:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)
        
    combined_path = os.path.join(output_folder, "combined", "combined_error_plot.png")
    plot_error_distribution(error_df, f"{study} Combined", combined_path, methods_order)
    
    combined_path_MMD = os.path.join(output_folder, "combined", "combined_MMD_plot.png")
    plot_MMD_distribution(
        data=error_df,title=f"{study} | Combined MMD",
        save_path=combined_path_MMD,
        methods_order=methods_order
    )
    
    stims = sorted(error_df["stim"].unique())
    populations_to_plot = sorted(error_df["population"].unique())
    markers_to_plot = sorted(error_df["marker"].unique())
    stims_to_plot = sorted(error_df["stim"].unique())
    
    #for stim in stims_to_plot:
        #for marker in markers_to_plot:
            #for pop in populations_to_plot:
                #subset = error_df[(error_df["stim"] == stim) &(error_df["marker"] == marker) &(error_df["population"] == pop)]
                #if subset.empty:
                    #continue
                #title = f"{study} | {stim} | {marker} | {pop}"
                #filename = f"{stim}_{marker}_{pop}.png".replace(" ", "_")
                #save_path = os.path.join(output_folder, "individual", filename)
                #plot_error_distribution(subset, title, save_path, methods_order)
                
                
                #title_mmd = f"{study} | {stim} | {marker} | {pop}"
                #filename_mmd = f"{stim}_{marker}_{pop}_MMD.png".replace(" ", "_")
                #save_path_mmd = os.path.join(output_folder, "individual", filename_mmd)
                #plot_MMD_distribution(data=subset,title=title_mmd,save_path=save_path_mmd,methods_order=methods_order)
                
    for stim in stims:
        subset = error_df[error_df["stim"] == stim]
        if subset.empty:
            continue
        title = f"{study} | Stim: {stim}"
        filename = f"{stim}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "stims", filename)
        plot_error_distribution(subset, title, save_path, methods_order)
        
        title_mmd = f"{study} | MMD per Stim: {stim}"
        filename_mmd = f"{stim}_MMD.png".replace(" ", "_")
        save_path_mmd = os.path.join(output_folder, "stims", filename_mmd)
        plot_MMD_distribution(
            data=subset,
            title=title_mmd,
            save_path=save_path_mmd,
            methods_order=methods_order
        )
        
    markers = sorted(error_df["marker"].unique())
    for marker in markers:
        subset = error_df[error_df["marker"] == marker]
        if subset.empty:
            continue
        title = f"{study} | Marker: {marker}"
        filename = f"{marker}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "markers", filename)
        plot_error_distribution(subset, title, save_path, methods_order)
        
        title_mmd = f"{study} | MMD per Marker: {marker}"
        filenam_mmd = f"{marker}_MMD.png".replace(" ", "_")
        save_path_mmd = os.path.join(output_folder, "markers", filenam_mmd)
        plot_MMD_distribution(
            data=subset,
            title=title_mmd,
            save_path=save_path_mmd,
            methods_order=methods_order
        )
        
    populations = sorted(error_df["population"].unique())
    for pop in populations:
        subset = error_df[error_df["population"] == pop]
        if subset.empty:
            continue
        title = f"{study} | Population: {pop}"
        filename = f"{pop}_error.png".replace(" ", "_")
        save_path = os.path.join(output_folder, "populations", filename)
        plot_error_distribution(subset, title, save_path, methods_order)
        
        title_mmd = f"{study} | MMD per Population: {pop}"
        filename_mmd = f"{pop}_MMD.png".replace(" ", "_")
        save_path_mmd = os.path.join(output_folder, "populations", filename_mmd)
        plot_MMD_distribution(
            data=subset,
            title=title_mmd,
            save_path=save_path_mmd,
            methods_order=methods_order
        )
        
    summary_plot_folder = os.path.join(output_folder, "summary_boxplots")
    os.makedirs(summary_plot_folder, exist_ok=True)
    
    median_stim_MMD = error_df.groupby(['method', 'stim'])['mmd'].median().reset_index()
    save_path_stim = os.path.join(summary_plot_folder, "boxplot_mmd_by_stim.png")
    plot_summary_metric_distribution(
        data=median_stim_MMD,
        metric="mmd",
        title=f"{study} | MMD per Stim",
        save_path=save_path_stim,
        methods_order=methods_order
    )

    
    median_pop_MMD = error_df.groupby(['method', 'population'])['mmd'].median().reset_index()
    save_path_pop = os.path.join(summary_plot_folder, "boxplot_mmd_by_population.png")
    plot_summary_metric_distribution(
        data=median_pop_MMD,
        metric="mmd",
        title=f"{study} | MMD per Population",
        save_path=save_path_pop,
        methods_order=methods_order
    )

   
    median_marker_MMD = error_df.groupby(['method', 'marker'])['mmd'].median().reset_index()
    save_path_marker = os.path.join(summary_plot_folder, "boxplot_mmd_by_marker.png")
    plot_summary_metric_distribution(
        data=median_marker_MMD,
        metric="mmd",
        title=f"{study} | MMD per Marker",
        save_path=save_path_marker,
        methods_order=methods_order
    )
    
    median_stim = error_df.groupby(['method', 'stim'])['error'].median().reset_index()
    save_path_stim = os.path.join(summary_plot_folder, "boxplot_median_by_stim.png")
    plot_summary_metric_distribution(
        data=median_stim,
        metric="error",
        title=f"{study} | Median Error per Stim",
        save_path=save_path_stim,
        methods_order=methods_order
    )

    
    median_pop = error_df.groupby(['method', 'population'])['error'].median().reset_index()
    save_path_pop = os.path.join(summary_plot_folder, "boxplot_median_by_population.png")
    plot_summary_metric_distribution(
        data=median_pop,
        metric="error",
        title=f"{study} | Median Error per Population",
        save_path=save_path_pop,
        methods_order=methods_order
    )

   
    median_marker = error_df.groupby(['method', 'marker'])['error'].median().reset_index()
    save_path_marker = os.path.join(summary_plot_folder, "boxplot_median_by_marker.png")
    plot_summary_metric_distribution(
        data=median_marker,
        metric="error",
        title=f"{study} | Median Error per Marker",
        save_path=save_path_marker,
        methods_order=methods_order
    )
    
    relative_stim = error_df.groupby(['method', 'stim']).apply(mae_relative).reset_index(name="error")
    save_path_rel_stim = os.path.join(summary_plot_folder, "boxplot_relative_error_by_stim.png")
    plot_summary_metric_distribution(
        data=relative_stim,
        metric="error",
        title=f"{study} | Relative Error per Stim",
        save_path=save_path_rel_stim,
        methods_order=methods_order
    )
    
    
    relative_pop = error_df.groupby(['method', 'population']).apply(mae_relative).reset_index(name="error")
    save_path_rel_pop = os.path.join(summary_plot_folder, "boxplot_relative_error_by_population.png")
    plot_summary_metric_distribution(
        data=relative_pop,
        metric="error",
        title=f"{study} | Relative Error per Population",
        save_path=save_path_rel_pop,
        methods_order=methods_order
    )

   
    relative_marker = error_df.groupby(['method', 'marker']).apply(mae_relative).reset_index(name="error")
    save_path_rel_marker = os.path.join(summary_plot_folder, "boxplot_relative_error_by_marker.png")
    plot_summary_metric_distribution(
        data=relative_marker,
        metric="error",
        title=f"{study} | Relative Error per Marker",
        save_path=save_path_rel_marker,
        methods_order=methods_order
    )
        
    correlation_plot_folder = os.path.join(output_folder, "correlation_boxplots")
    os.makedirs(correlation_plot_folder, exist_ok=True)

    correlation_df = compute_correlations(error_df, group_cols=["stim"])
    plot_correlation_boxplots(
        correlation_df, metric="spearman", group_name="stim",
        methods_order=methods_order,
        out_dir=correlation_plot_folder,
        title=f"{study} | Spearman group by Stim"
    )
    plot_correlation_boxplots(
        correlation_df, metric="pearson", group_name="stim",
        methods_order=methods_order,
        out_dir=correlation_plot_folder,
        title=f"{study} | Pearson group by Stim"
    )

    correlation_df = compute_correlations(error_df, group_cols=["population"])
    plot_correlation_boxplots(
        correlation_df, metric="spearman", group_name="population",
        methods_order=methods_order,
        out_dir=correlation_plot_folder,
        title=f"{study} | Spearman group by Population"
    )
    plot_correlation_boxplots(
        correlation_df, metric="pearson", group_name="population",
        methods_order=methods_order,
        out_dir=correlation_plot_folder,
        title=f"{study} | Pearson group by Population"
    )

    correlation_df = compute_correlations(error_df, group_cols=["marker"])
    plot_correlation_boxplots(
        correlation_df, metric="spearman", group_name="marker",
        methods_order=methods_order,
        out_dir=correlation_plot_folder,
        title=f"{study} | Spearman group by Marker"
    )
    plot_correlation_boxplots(
        correlation_df, metric="pearson", group_name="marker",
        methods_order=methods_order,
        out_dir=correlation_plot_folder,
        title=f"{study} | Pearson group by Marker"
    )
    
    group_cols = ["stim", "population", "marker"]
    corr_df = compute_correlations(error_df, group_cols)
    
    plot_correlation_boxplots(corr_df, metric="spearman", group_name="stim_CT_marker",
                              methods_order=methods_order, out_dir=correlation_plot_folder,
                              title="Spearman correlation by Stim x CT x Marker")
    
    plot_correlation_boxplots(corr_df, metric="pearson", group_name="stim_CT_marker",
                              methods_order=methods_order, out_dir=correlation_plot_folder,
                              title="Pearson correlation by Stim x CT x Marker")
    for stim in corr_df["stim"].unique():
        sub_df = corr_df[corr_df["stim"] == stim]
        plot_correlation_boxplots(sub_df, metric="spearman", group_name=f"stim_{stim}",
                                  methods_order=methods_order, out_dir=correlation_plot_folder,
                                  title=f"Spearman Correlation by Stim x CT x Marker- Stim: {stim}")
    
    for marker in corr_df["marker"].unique():
        sub_df = corr_df[corr_df["marker"] == marker]
        plot_correlation_boxplots(sub_df, metric="spearman", group_name=f"marker_{marker}",
                                  methods_order=methods_order, out_dir=correlation_plot_folder,
                                  title=f"Spearman Correlation by Stim x CT x Marker- Marker: {marker}")
    for ct in corr_df["population"].unique():
        sub_df = corr_df[corr_df["population"] == ct]
        plot_correlation_boxplots(sub_df, metric="spearman", group_name=f"CT_{ct}",
                                  methods_order=methods_order, out_dir=correlation_plot_folder,
                                  title=f"Spearman Correlation by Stim x CT x Marker- Cell Type: {ct}")




def mae_relative(group):
        abs_error = np.abs(group["error"])
        abs_true = np.abs(group["true"])
        return np.mean(abs_error / (abs_true + 1e-8))


def compute_summary_metrics(error_df):
    summary = error_df.groupby(["stim", "marker", "population", "method"]).agg(
                MAE=("error", "mean"),
                RMSE=("error", lambda x: np.sqrt(np.mean(x**2))),
                MAE_relative=("true", lambda x: np.mean(np.abs(error_df.loc[x.index, "error"])) / (np.mean(np.abs(x)) + 1e-8))
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
FOLD_INFO_FILE = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/ptb_cellwise_variance_cv_fold_info.csv"
fold_info = defaultdict(lambda: defaultdict(dict)) # Structure: fold_info[stim][sanitized_celltype][fold_index] = [test_patient1, ...]
stim_celltype_pairs_in_folds = set() # Keep track of (stim, original_celltype) pairs processed
try:
    with open(FOLD_INFO_FILE, 'r') as f:
        header = f.readline().strip().split(',') # Read header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4: continue # stim, sanitized, original, fold, patient1...
            stim, sanitized_ct, original_ct, fold_idx_str = parts[:4]
            test_patients = parts[4:]
            try:
                fold_idx = int(fold_idx_str)
                fold_info[stim][sanitized_ct][fold_idx] = test_patients
                stim_celltype_pairs_in_folds.add((stim, original_ct))
            except ValueError:
                print(f"[WARN] Invalid fold index '{fold_idx_str}' in line: {line.strip()}")
    print(f"Loaded fold info for {len(fold_info)} stims.")
    print(f"Total (stim, original_celltype) pairs with fold info: {len(stim_celltype_pairs_in_folds)}")
except FileNotFoundError:
    sys.exit(f"FATAL ERROR: Fold info file not found: {FOLD_INFO_FILE}")
except Exception as e:
    sys.exit(f"FATAL ERROR: Failed to read fold info file: {e}")

stim_list=['PI','TNFa','IFNa','IL33','IL246','LPS','GMCSF']
def main():
    ptb_errors_global = []

    for stim in stim_list:
        out_dir = f"/home/groups/gbrice/ptb-drugscreen/ot/cellot/cellwise"
        base_dir = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/cellwise/combined_plots_models_20markers/"

        os.makedirs(out_dir, exist_ok=True)
        cohort='ptb-drugscreen'
        # Use combined_transformed paths
        combined_dir = out_dir
        
        #true_path_olivier = os.path.join(combined_dir, f"cross_validation_olivier_2/{cohort}_{stim}_transformed3.csv")
        true_path_different_IO = os.path.join(combined_dir, f"cross_validation_different_IO/{cohort}_{stim}_transformed.csv")
        #true_path_shuffled_20marks = os.path.join(combined_dir, f"cross_validation_shuffled_20marks/{cohort}_{stim}_transformed.csv")
        true_path_original_20marks = os.path.join(combined_dir, f"cross_validation_original_20marks/{cohort}_{stim}_transformed.csv")
        true_path_original_median = os.path.join(combined_dir, f"cross_validation_original_median_20marks/{cohort}_{stim}_transformed.csv")
        
        #ot_path_olivier = os.path.join(combined_dir, f"cross_validation_olivier_2/{cohort}_{stim}_predicted_transformed3.csv")
        ot_path_different_IO = os.path.join(combined_dir, f"cross_validation_different_IO/{cohort}_{stim}_predicted_transformed.csv")
        #ot_path_shuffled_20marks = os.path.join(combined_dir, f"cross_validation_shuffled_20marks/{cohort}_{stim}_predicted_transformed.csv")
        ot_path_original_20marks = os.path.join(combined_dir, f"cross_validation_original_20marks/{cohort}_{stim}_predicted_transformed.csv")
        ot_path_original_median = os.path.join(combined_dir, f"cross_validation_original_median_20marks/{cohort}_{stim}_predicted_transformed.csv")
        
        #olivier_ot_errors = compute_ot_errors(true_path_olivier, ot_path_olivier,'Olivier')
        different_IO_ot_errors = compute_ot_errors(true_path_different_IO, ot_path_different_IO,'Semishuffled')
        #shuffled_20marks_ot_errors = compute_ot_errors(true_path_shuffled_20marks, ot_path_shuffled_20marks,'Shuffled')
        original_20marks_ot_errors = compute_ot_errors(true_path_original_20marks, ot_path_original_20marks,'Original')
        original_median_ot_errors = compute_ot_errors(true_path_original_median, ot_path_original_median,'original_median')
        
        #ptb_errors = pd.concat([olivier_ot_errors, original_ot_errors,original_20marks_ot_errors,OG_39m_ot_errors], ignore_index=True)
        ptb_errors = pd.concat([different_IO_ot_errors,original_20marks_ot_errors,original_median_ot_errors], ignore_index=True)
        
        ptb_errors_filtered=ptb_errors
        ptb_errors_filtered["stim"] = stim
        ptb_errors_global.append(ptb_errors_filtered)

        plots_dir = base_dir+ f"{cohort}_{stim}/"
        #os.makedirs(plots_dir, exist_ok=True)
        
        # Define method order.
        #ptb_methods_order = ["Olivier", "Original",'original_20marks','39m_OG']
        ptb_methods_order = ['Semishuffled','Original','original_median']
        
        #generate_plots(ptb_errors_filtered, os.path.join(plots_dir, "ptb"), ptb_methods_order, "Ptb")
        
        #ptb_summary = compute_summary_metrics(ptb_errors_filtered)
        
        # Create output folders for summary plots under combined_plots.
        ptb_mae_dir = os.path.join(plots_dir, "ptb_MAE_plots")
        ptb_rmse_dir = os.path.join(plots_dir, "ptb_RMSE_plots")
        ptb_relative_dir = os.path.join(plots_dir, "ptb_MAE_relative_plots")

        #os.makedirs(ptb_mae_dir, exist_ok=True)
        #os.makedirs(ptb_rmse_dir, exist_ok=True)
        #os.makedirs(ptb_relative_dir, exist_ok=True)

        #generate_summary_plots(ptb_summary, ptb_mae_dir, "MAE", ptb_methods_order, "Ptb MAE")
        #generate_summary_plots(ptb_summary, ptb_rmse_dir, "RMSE", ptb_methods_order, "Ptb RMSE")
        #generate_summary_plots(ptb_summary, ptb_relative_dir, "MAE_relative", ptb_methods_order, "Ptb MAE Relative")
    
    ptb_errors_global_df = pd.concat(ptb_errors_global, ignore_index=True)
    

    global_plots_dir = base_dir
    os.makedirs(global_plots_dir, exist_ok=True)
    
    #ptb_methods_order = ["Olivier", "Original",'original_20marks','39m_OG']
    #ptb_methods_order = ['original_20marks','original_median']
    generate_plots(ptb_errors_global_df, os.path.join(global_plots_dir, "ptb_all"), ptb_methods_order, "Ptb (All Stims)")
    
    ptb_global_summary = compute_summary_metrics(ptb_errors_global_df)

    ptb_global_mae_dir = os.path.join(global_plots_dir, "ptb_MAE_plots")
    ptb_global_rmse_dir = os.path.join(global_plots_dir, "ptb_RMSE_plots")
    ptb_global_relative_dir = os.path.join(global_plots_dir, "ptb_MAE_relative_plots")
    
    os.makedirs(ptb_global_mae_dir, exist_ok=True)
    os.makedirs(ptb_global_rmse_dir, exist_ok=True)
    os.makedirs(ptb_global_relative_dir, exist_ok=True)
    
    generate_summary_plots(ptb_global_summary, ptb_global_mae_dir, "MAE", ptb_methods_order, "Ptb Global MAE")
    generate_summary_plots(ptb_global_summary, ptb_global_rmse_dir, "RMSE", ptb_methods_order, "Ptb Global RMSE")
    generate_summary_plots(ptb_global_summary, ptb_global_relative_dir, "MAE_relative", ptb_methods_order, "Ptb MAE Relative")
    

    print("Summary plots generated.")
    df = ptb_errors_global_df.copy()


    grouped = df.groupby(["method","stim", "population", "marker"]).apply(
        lambda g: pd.Series({
            "MAE": mean_absolute_error(g["true"], g["pred"]),
            "RMSE": mean_squared_error(g["true"], g["pred"]),
            "R2": r2_score(g["true"], g["pred"]) if len(g["true"]) > 1 else np.nan,
            "RelativeError": np.median(np.abs(g["true"] - g["pred"]) / (np.abs(g["true"]) + 1e-6))
        })
    ).reset_index()
    
    summary = grouped.groupby(["method","stim"])[["MAE", "RMSE"]].median().reset_index()
    
    original_mae = summary[summary["method"] == "original_20marks"][["stim", "MAE"]]
    original_mae = original_mae.rename(columns={"MAE": "MAE_original"})
    
    summary = summary.merge(original_mae, on="stim", how="left")
    summary["MAE"] = summary["MAE"].astype(float)
    original_mae["MAE_original"] = original_mae["MAE_original"].astype(float)

    summary["MAE_relative_variation"] = (summary["MAE"] - summary["MAE_original"]) / (summary["MAE_original"] + 1e-8)
    summary = summary[["method","stim","MAE", "RMSE","MAE_relative_variation"]]
    ptb_global_mae_dir = os.path.join(global_plots_dir, "ptb_error_summary_by_stim.csv")
    summary.to_csv(ptb_global_mae_dir, index=False)
    print(f"finished for the error summary per stim csv, saved at {ptb_global_mae_dir}")
    
    detailed_summary = grouped.copy()
    
    detailed_summary = detailed_summary[["method","stim", "population", "marker", "MAE", "RMSE"]]

    ptb_detailed_error_path = os.path.join(global_plots_dir, "ptb_detailed_error_by_stim_CT_marker.csv")
    detailed_summary.to_csv(ptb_detailed_error_path, index=False)
    
    print(f"Finished detailed error table by stim x CT x marker, saved at {ptb_detailed_error_path}")


if __name__ == "__main__":
    main()