# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:22:07 2025

@author: kobas
"""

# -*- coding: utf-8 -*-
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from matplotlib.font_manager import FontProperties

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore

# ---------------------------
# Helpers
# ---------------------------

def filter_outliers_zscore(df, column, threshold=3):
    """Z-score filter within each (part_id × scenario) group, keeping NaNs."""
    if column in df.columns:
        df = (
            df.groupby(['part_id', 'scenario'], group_keys=False)
              .apply(lambda g: g[(np.abs(zscore(g[column], nan_policy='omit')) <= threshold) | g[column].isna()])
        )
    return df

# ---------------------------
# Paths & fonts
# ---------------------------

# try:
#     ROOT = Path(__file__).resolve().parents[1]  # repo root if running as a script
# except NameError:
#     ROOT = Path.cwd()  # fallback if running in an interactive console/notebook

ROOT = Path(r"C:\Users\kobas\OneDrive\Desktop\Meta\SL2201 PMV\Git")

DATA   = ROOT / "01 Data"
GRAPHS = ROOT / "03 Graphs"
OUT    = ROOT / "04 Model Outputs/06 Timeseries"
FONTS  = ROOT / "Fonts"

GRAPHS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Optional custom font
warnings.filterwarnings("ignore", message=".*findfont:.*")
try:
    font_path = FONTS / "Archivo_SemiCondensed-Medium.ttf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        _custom_font = FontProperties(fname=str(font_path))
        rcParams["font.family"] = _custom_font.get_name()
except Exception:
    pass

# Embed editable text in vector outputs
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype']  = 42

# ---------------------------
# Styling & config
# ---------------------------
font_smaller = 6
font_small   = 7
font_larger  = 8

# Scenario colors
scenario_colors = {
    'T1': '#22181C',
    'T2': '#59C3C3',
    'T3': '#FFAC27',
    'T4': '#EA002D'
}

# Clusters & scenarios
clusters  = ['Morning', 'Afternoon']
scenarios = ["T1", "T2", "T3", "T4"]

# Variables to plot
variables = ['skin_temp', 'EDA_Tonic', 'hr_clean', 'rmssd', 'lf_hf']

# Titles (you referred to variable_titles; define that alias)
variable_titles = {
    'skin_temp': "Skin temperature (°C)",
    'EDA_Tonic': "EDA tonic (µS)",
    'hr_clean':  "Heart rate (bpm)",
    'rmssd':     "RMSSD (ms)",
    'lf_hf':     "LF/HF ratio",
}

# Significance/contrast display rules
d_threshold = 0.45   # write contrasts in table when |d| > 0.45 (regardless of p)
alpha_fdr   = 0.05   # FDR control level

# ---------------------------
# 1) Import & filter data
# ---------------------------

csv_path = DATA / '00_SL2201_CombinedData.csv'
if not csv_path.exists():
    raise FileNotFoundError(f"Data file not found: {csv_path}")

data_full = pd.read_csv(str(csv_path))

# keep only relevant runs
data = data_full[
    (data_full['lab-id'] == 'SL') &
    (data_full['break'] != 1)
].copy()

# rename columns
data = data.rename(columns={
    "time-elapsed": "time_elapsed",
    "lf/hf": "lf_hf",
    "part-id": "part_id",
    "hr-clean": "hr_clean",
    "skin-temp-wrist-smooth": "skin_temp"
})

# filter 2–178 minutes
data = data[(data['time_elapsed'] >= 2) & (data['time_elapsed'] < 179)].copy()

# compute 10-minute bins
data['time_bin'] = (data['time_elapsed'] // 10) * 10

# drop rows with missing critical vars
data = data.dropna(subset=['part_id', 'skin_temp', 'time_bin', 'hr_clean', 'lf_hf', 'rmssd', 'EDA_Tonic']).copy()

# categoricals with explicit order
data['scenario'] = pd.Categorical(data['scenario'], categories=scenarios, ordered=True)
data['cluster']  = pd.Categorical(data['cluster'],  categories=clusters,  ordered=True)
data['part_id']  = data['part_id'].astype('category')

# Outlier removal per var
for v in variables:
    before = len(data)
    data = filter_outliers_zscore(data, v, threshold=3).copy()
    after = len(data)
    print(f"Rows after Z-score filtering for {v}: {after} (removed {before - after})")

# Build the set of all 10-min bins present (sorted ints). Ensures first bin is the reference.
all_time_bins = sorted(pd.unique(data['time_bin'].astype(int)))
if len(all_time_bins) == 0:
    raise ValueError("No time bins found after filtering.")

# Keep a clean copy for modeling (if needed later)
data_for_model = data.copy()

#%% Plot: Time × Sex × Cluster with per-scenario FDR & d>0.44 table entries (gray bg if FDR-sig)
# styling knobs
MARKERSIZE   = 3
CAPSIZE      = 1
ELINEWIDTH   = 1
CAPTHICK     = 1
LINEWIDTH    = 0.5
GRID_ALPHA   = 0.5
GRID_LW      = 0.5
CELL_GRAY    = "0.92"  # base cell bg
SIG_GRAY     = "0.9"  # significant cell bg (medium)
AXH0_LW      = 0.5
AXH0_COLOR   = "black"
VTICKS       = (60, 120)  # vertical guide lines like in your figure; set () to disable

# per-variable y-lims and grid step (fill in as needed)
gridline_steps = {
    'skin_temp': 0.5,
    'EDA_Tonic': 1,
    'hr_clean': 5,
    'rmssd': 20,
    'lf_hf': 0.2,
}
variables_all = {
    'hr_clean':   {'ylim': (-15, 15)},
    'rmssd':      {'ylim': (-60, 80)},
    'lf_hf':      {'ylim': (-0.4, 0.6)},
    'skin_temp':  {'ylim': ( -1.75,  1.75)},   
    'EDA_Tonic':  {'ylim': ( -2.0, 5.0)},
}

# Make sure sex has a stable order
if 'sex' in data.columns:
    desired_sex_order = [s for s in ['Female', 'Male'] if s in data['sex'].unique().tolist()]
    sex_values = desired_sex_order if desired_sex_order else list(pd.Index(data['sex'].dropna().unique()))
else:
    raise KeyError("Column 'sex' not found in data.")

unique_clusters  = list(pd.Index(data['cluster'].dropna().unique()))
unique_scenarios = list(pd.Index(data['scenario'].dropna().unique()))

excel_rows = []  # collect contrasts for audit
for variable in variables:
    print(f"\nProcessing variable: {variable}")

    # figure & spacing: add bottom room for tables
    fig, axes = plt.subplots(2, 2, figsize=(24 / 2.54, 16 / 2.54), sharey=True)

    
    # Carve per-axes space for tables: lift axes and shrink height a bit
    for r in range(axes.shape[0]):
        for c in range(axes.shape[1]):
            ax = axes[r, c]
            x0, y0, w, h = ax.get_position().bounds
            # Top row needs more room for its table (so it doesn't hit row 2 plots)
            pad = 0.09 if r == 0 else 0.07
            ax.set_position([x0, y0 + pad, w, h - pad])
    
    # Use a slightly shallower table for the TOP row, deeper for BOTTOM row
    bbox_top    = [0.02, -0.65, 0.96, 0.45]  # (x0, y0, w, h) in axes coords
    bbox_bottom = [0.02, -0.65, 0.96, 0.45]

    fig.suptitle(f"{variable_titles.get(variable, variable)} by time cluster and sex",
                 fontsize=font_larger, y=0.95)
    plt.subplots_adjust(hspace=0.9, bottom=0.35, top=0.9, wspace=0.1)

    # y-lims + grid step
    ylim = variables_all.get(variable, {}).get('ylim', None)
    ystep = gridline_steps.get(variable, None)

    for row_idx, sex_value in enumerate(sex_values):
        subset_data = data.loc[data['sex'] == sex_value].copy()

        for col_idx, cluster_value in enumerate(unique_clusters):
            ax = axes[row_idx, col_idx]

            cluster_subset = subset_data.loc[subset_data['cluster'] == cluster_value].copy()
            cluster_subset['time_bin'] = pd.Categorical(cluster_subset['time_bin'],
                                                        categories=all_time_bins, ordered=True)
            cluster_subset = cluster_subset.dropna(subset=['part_id', variable, 'time_bin']).copy()
            if cluster_subset.empty:
                ax.set_title(f"Sex: {sex_value}, Cluster: {cluster_value}", fontsize=font_small)
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            # shared residual variance per subgroup (sex×cluster)
            combined_model = smf.mixedlm(
                f"{variable} ~ C(time_bin)",
                cluster_subset,
                groups=cluster_subset["part_id"],
                re_formula="~1"
            )
            combined_result = combined_model.fit()
            ref_variance = float(getattr(combined_result, "scale", np.nan))

            scenario_table_data = []  # rows = scenarios; cells = (text, sig_bool)
            for scenario in unique_scenarios:
                scenario_data = cluster_subset.loc[cluster_subset['scenario'] == scenario].copy()
                if scenario_data.empty:
                    scenario_table_data.append([("", False)] * len(all_time_bins))
                    continue

                scenario_data['time_bin'] = pd.Categorical(scenario_data['time_bin'],
                                                           categories=all_time_bins, ordered=True)
                scenario_data = scenario_data.dropna(subset=['part_id', variable, 'time_bin']).copy()
                if scenario_data.empty:
                    scenario_table_data.append([("", False)] * len(all_time_bins))
                    continue

                # per-scenario model
                individual_model = smf.mixedlm(
                    f"{variable} ~ C(time_bin)",
                    scenario_data,
                    groups=scenario_data["part_id"],
                    re_formula="~1"
                )
                individual_result = individual_model.fit()

                # FDR over non-reference time bins
                coeff_keys, p_values = [], []
                for tb in all_time_bins[1:]:
                    k = f"C(time_bin)[T.{tb}]"
                    if k in individual_result.pvalues.index:
                        coeff_keys.append(k)
                        p_values.append(individual_result.pvalues[k])
                pval_map = {}
                if p_values:
                    _, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha_fdr, method='fdr_bh')
                    pval_map = dict(zip(coeff_keys, pvals_corrected))

                # plot arrays + row
                time_bins_plot, coefs, ci_lo, ci_hi = [], [], [], []
                row_cells = [("", False)]  # reference col (blank)
                conf_int = individual_result.conf_int()
                ref_tb = int(all_time_bins[0])

                for tb in all_time_bins[1:]:
                    k = f"C(time_bin)[T.{tb}]"
                    coef = individual_result.params.get(k, np.nan)
                    text = ""
                    sig  = False
                    lb = ub = np.nan

                    if pd.notna(coef):
                        d_val = (float(coef) / np.sqrt(ref_variance)) if (pd.notna(ref_variance) and ref_variance > 0) else np.nan
                        p_raw = individual_result.pvalues.get(k, np.nan)
                        fdr_p = pval_map.get(k, np.nan)
                        sig   = (pd.notna(fdr_p) and fdr_p < alpha_fdr)

                        # write ONLY if |d| ≥ threshold
                        if pd.notna(d_val) and abs(d_val) >= d_threshold:
                            text = f"{d_val:.2f}"

                        # CIs + plotting
                        if k in conf_int.index:
                            lb = float(conf_int.loc[k, 0]); ub = float(conf_int.loc[k, 1])
                            time_bins_plot.append(int(tb))
                            coefs.append(float(coef)); ci_lo.append(lb); ci_hi.append(ub)

                        # collect for Excel
                        excel_rows.append({
                            "variable": variable, "sex": sex_value, "cluster": cluster_value, "scenario": scenario,
                            "time_bin": int(tb), "coef_delta_vs_ref": float(coef),
                            "ci_low": lb, "ci_high": ub, "cohens_d": d_val,
                            "p_raw": p_raw, "p_fdr": fdr_p, "fdr_sig": bool(sig),
                            "ref_variance": ref_variance
                        })

                    row_cells.append((text, sig))

                # pad row to full width
                if len(row_cells) < len(all_time_bins):
                    row_cells += [("", False)] * (len(all_time_bins) - len(row_cells))
                scenario_table_data.append(row_cells)

                # baseline anchor & plot
                time_bins_plot.append(ref_tb); coefs.append(0.0); ci_lo.append(0.0); ci_hi.append(0.0)
                order = np.argsort(time_bins_plot)
                time_bins_plot = list(np.array(time_bins_plot)[order])
                coefs = list(np.array(coefs)[order]); ci_lo = list(np.array(ci_lo)[order]); ci_hi = list(np.array(ci_hi)[order])

                yerr = [np.array(coefs) - np.array(ci_lo), np.array(ci_hi) - np.array(coefs)]
                ax.errorbar(
                    time_bins_plot, coefs, yerr=yerr,
                    fmt='o',
                    markersize=MARKERSIZE,
                    capsize=CAPSIZE,
                    elinewidth=ELINEWIDTH,
                    capthick=CAPTHICK,
                    label=f"{scenario}",
                    color=scenario_colors.get(scenario, '#000000')
                )
                ax.plot(
                    time_bins_plot, coefs,
                    linestyle='-',
                    linewidth=LINEWIDTH,
                    color=scenario_colors.get(scenario, '#000000')
                )

            # ---- table: text only, then tint cells ----
            cell_text_only = [[t for (t, s) in row] for row in scenario_table_data]
            table_obj = ax.table(
                cellText=cell_text_only,
                rowLabels=unique_scenarios,
                colLabels=None,
                loc='bottom',
                bbox=(bbox_top if row_idx == 0 else bbox_bottom)
            )
            table_obj.auto_set_font_size(False)
            table_obj.set_fontsize(font_smaller)

            # set base bg + center text
            for (r, c), cell in table_obj.get_celld().items():
                cell.set_text_props(ha='center', va='center', usetex=False)
                text = cell.get_text().get_text().strip()
                if text != "":                # only shade if there's something written
                    cell.set_facecolor("white")
                else:
                    cell.set_facecolor("white")  # leave empty ones white
                cell.set_edgecolor('0.7')
                cell.set_linewidth(0.6)


            # shade significant cells darker
            n_rows = len(scenario_table_data); n_cols = len(all_time_bins)
            for r in range(n_rows):
                for c in range(n_cols):
                    text, sig = scenario_table_data[r][c]
                    if sig and text != "":
                        cell = table_obj.get_celld().get((r, c))
                        if cell is not None:
                            cell.set_facecolor(SIG_GRAY)

            # cosmetics: baseline & grid style like your figure
            ax.axhline(y=0, color=AXH0_COLOR, linestyle="-", linewidth=AXH0_LW)
            if VTICKS:
                for vx in VTICKS:
                    ax.axvline(vx, color='0.2', linewidth=0.5)

            # y-lims + grid ticks
            if ylim:
                ax.set_ylim(*ylim)
            if ystep:
                y0, y1 = ax.get_ylim()
                ax.set_yticks(np.arange(np.floor(y0/ystep)*ystep, np.ceil(y1/ystep)*ystep + 1e-9, ystep))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f"{sex_value} — {cluster_value}", fontsize=font_small)
            # ax.set_xlabel("Elapsed time (minutes)", fontsize=font_small)
            ax.tick_params(axis='y', labelsize=font_small)
            ax.set_xticks(all_time_bins)
            ax.set_xticklabels([f"{t}" for t in all_time_bins], fontsize=font_small)

            # light grid
            ax.grid(which='major', axis='both', alpha=GRID_ALPHA, linewidth=GRID_LW)

    # legend
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.15),  # closer to plots than 0.00
            ncol=max(1, len(unique_scenarios)),
            fontsize=font_small,
            frameon=False
        )


    # save (avoid tight_layout after tables)
    out_pdf = OUT / f"{variable}_cluster_sex_plot.pdf"
    out_png = OUT / f"{variable}_cluster_sex_plot.png"
    plt.savefig(out_pdf, format="pdf", dpi=600, bbox_inches='tight')
    plt.savefig(out_png, format="png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

print("\nDone.")



# ===== Export all contrasts to Excel (one flat table) =====
if excel_rows:
    contrasts_df = pd.DataFrame(excel_rows)
    xlsx_path = OUT / "contrasts_time_sex_cluster_by_scenario.xlsx"
    csv_path  = OUT / "contrasts_time_sex_cluster_by_scenario.csv"

    # Try Excel first
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            # Option 1: single sheet
            contrasts_df.to_excel(writer, index=False, sheet_name="all_contrasts")

            # Option 2 (optional): per-variable sheets
            # for v, dfv in contrasts_df.groupby("variable"):
            #     dfv.to_excel(writer, index=False, sheet_name=str(v)[:31])  # Excel sheet name limit
        print(f"Saved Excel: {xlsx_path}")
    except Exception as e:
        print(f"Excel export failed ({e}). Saving CSV instead.")
        contrasts_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
else:
    print("No contrasts collected; nothing to export.")
