# -*- coding: utf-8 -*-
"""
Enhanced version of 01_MixedEffects_looped.py with rich EXPORTS
All outputs are stored in a dedicated folder: 01_Model_Outputs_Baseline

Exports per variable:
- CSV of fixed-effects table (coef, SE, z, p, CI)
- DOCX summary
- PNG Q-Q plot
- SVG & JPEG errorbar plot
- CSV of all pairwise SCENARIO contrasts (T1–T4), with:
    estimate, SE, z, raw p, q (FDR), Cohen's d, 95% CI

Also:
- Master CSV concatenating all variables' contrasts
- Master Excel workbook with one sheet per variable (contrasts)
- Combined multi-panel plot (PNG + JPEG)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from docx import Document
import scipy.stats as stats
from scipy.stats import zscore
import matplotlib.ticker as ticker
from statsmodels.stats.multitest import multipletests
import os
from pathlib import Path

# ------------------------ Output directory ------------------------
output_dir = "./04 Model Outputs/01_Model_Outputs_Baseline"
os.makedirs(output_dir, exist_ok=True)

# ------------------------ Helper functions ------------------------

def filter_outliers_iqr(df, variable):
    def iqr_outliers(group):
        Q1 = group[variable].quantile(0.25)
        Q3 = group[variable].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return group[(group[variable] >= lower_bound) & (group[variable] <= upper_bound)]
    return df.groupby(['part_id', 'scenario'], group_keys=False).apply(iqr_outliers)

def filter_outliers_zscore(df, variable, threshold=3):
    def zscore_outliers(group):
        group = group.copy()
        group['zscore'] = zscore(group[variable], nan_policy='omit')
        return group[(group['zscore'] <= threshold) & (group['zscore'] >= -threshold)]
    filtered_df = df.groupby(['part_id', 'scenario'], group_keys=False).apply(zscore_outliers)
    return filtered_df.drop(columns='zscore', errors='ignore')

def normalize_non_normal_data(df, variable):
    return np.log1p(df[variable])

def resample_within_groups(group):
    group = group.set_index('datetime')
    numeric_columns = group.select_dtypes(include=[np.number]).columns
    resampled_group = group[numeric_columns].resample('5T').mean()
    resampled_group['scenario'] = group['scenario'].resample('5T').first()
    resampled_group['part_id'] = group['part_id'].resample('5T').first()
    resampled_group = resampled_group.reset_index()
    return resampled_group

def calculate_cohens_d_groups(group1, group2):
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(), group2.std()
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean2 - mean1) / pooled_std if pooled_std > 0 else np.nan

def cohens_d_from_estimate(est, resid_var):
    return est / np.sqrt(resid_var) if resid_var > 0 else np.nan

def mean_vector_for_scenario(param_index, scenario):
    v = pd.Series(0.0, index=param_index)
    if 'Intercept' in v.index:
        v['Intercept'] = 1.0
    if scenario in ['T2', 'T3', 'T4']:
        key = f"scenario[T.{scenario}]"
        if key in v.index:
            v[key] = 1.0
    return v

def contrast_from_fit(result, A, B):
    fe = result.fe_params
    cov = result.cov_params().loc[fe.index, fe.index]
    vA = mean_vector_for_scenario(fe.index, A)
    vB = mean_vector_for_scenario(fe.index, B)
    c = (vA - vB)
    est = float(np.dot(c.values, fe.values))
    var = float(c.values @ cov.values @ c.values)
    se = np.sqrt(var) if var >= 0 else np.nan
    z = est / se if (se is not None and se > 0) else np.nan
    p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    ci_lo = est - 1.96 * se if np.isfinite(se) else np.nan
    ci_hi = est + 1.96 * se if np.isfinite(se) else np.nan
    d = cohens_d_from_estimate(est, result.scale)
    return est, se, z, p, d, ci_lo, ci_hi

def draw_cohens_d_brackets(ax, df_contrasts, x_positions, y_base, spacing, yrange):
    """
    Apply 3 rules to draw brackets + labels for Cohen's d:

      1) |d| < 0.45 and q > 0.05  -> DROP (not plotted, no space)
      2) |d| >= 0.45 and q > 0.05 -> gray bracket + gray text (normal)
      3) |d| >= 0.45 and q < 0.05 -> black bracket + black text (normal)
    """
    # cap = 0.02 * yrange
    text_pad = 0.01 * yrange

    # standard ordering of pairs
    all_pairs = [('T1','T2'), ('T1','T3'), ('T1','T4'),
                 ('T2','T3'), ('T2','T4'), ('T3','T4')]

    # filter pairs based on rules
    filtered = []
    for (A, B) in all_pairs:
        row = df_contrasts[(df_contrasts['level_A'] == A) &
                           (df_contrasts['level_B'] == B)]
        if row.empty:
            continue
        row = row.iloc[0]
        d = row.get('cohens_d', np.nan)
        q = row.get('q', np.nan)

        if pd.isna(d):
            continue
        # Rule 1 → skip completely
        if abs(d) < 0.45 and (pd.isna(q) or q > 0.05):
            continue
        # Rule 2 → gray
        elif abs(d) >= 0.45 and (pd.isna(q) or q > 0.05):
            filtered.append((A, B, d, q, 'gray'))
        # Rule 3 → black
        elif abs(d) >= 0.45 and q <= 0.05:
            filtered.append((A, B, d, q, 'black'))

    # now actually plot them, stacked with spacing
    for i, (A, B, d, q, color) in enumerate(filtered):
        y = y_base + i * spacing
        x1, x2 = x_positions[A], x_positions[B]

        # bracket
        ax.plot([x1, x1, x2, x2],
                [y, y, y, y],
                lw=0.8, color=color, clip_on=False)

        # label
        label = f"{d:.2f}"
        ax.text((x1 + x2) / 2.0, y + text_pad, label,
                ha="center", va="bottom",
                fontsize=7, fontweight='normal',
                color=color, clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.8))


# ------------------------ Config ------------------------


# ==== Repo-relative paths (no private absolute paths) ====
ROOT = Path(__file__).resolve().parents[1]  # repo root (assuming scripts live in 02_Codes/)
DATA   = ROOT / "01 Data"
GRAPHS = ROOT / "03 Graphs"
OUT    = ROOT / "04 Model_Outputs"
FONTS  = ROOT / "Fonts"

# Optional custom font
try:
    from matplotlib import rcParams
    from matplotlib import font_manager
    font_path = FONTS / "Archivo_SemiCondensed-Medium.ttf"
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        from matplotlib.font_manager import FontProperties
        _custom_font = FontProperties(fname=str(font_path))
        rcParams["font.family"] = _custom_font.get_name()
except Exception:
    pass

# Embed editable text in vector outputs
try:
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype']  = 42
except Exception:
    pass



scenario_colors = {
    'T1': '#22181C',
    'T2': '#59C3C3',
    'T3': '#FFAC27',
    'T4': '#EA002D'
}

downsample_variables = ['lf_hf', 'rmssd', 'EDA_Tonic', 'skin_temp_wrist_smooth']
original_resolution_variables = ['hr_clean']

variables = {
    'skin_temp_wrist_smooth': {'ylim': (33, 38), 'spacing': 0.5, 'start_offset': -0.18},
    'EDA_Tonic': {'ylim': (0, 11), 'spacing': 1.15, 'start_offset': -0.7},
    'hr_clean': {'ylim': (74, 88), 'spacing': 1.5, 'start_offset': 1},
    'rmssd': {'ylim': (100, 220), 'spacing': 13, 'start_offset': 0},
    'lf_hf': {'ylim': (0.4, 0.9), 'spacing': 0.03, 'start_offset': -0.01}
}

gridline_steps = {'skin_temp_wrist_smooth': 1, 'EDA_Tonic': 1.0, 'hr_clean': 2.0, 'rmssd': 20.0, 'lf_hf': 0.1}

variable_titles = {
    'skin_temp_wrist_smooth': "Skin temperature (°C)",
    'EDA_Tonic': "EDA tonic (µS)",
    'hr_clean': "Heart rate (bpm)",
    'rmssd': "RMSSD (ms)",
    'lf_hf': "LF/HF ratio",
}

output_prefix = "analysis_output"

# ------------------------ Load & preprocess ------------------------

data = pd.read_csv(str(DATA / '00_SL2201_CombinedData.csv'))
filtered_data = data[(data['lab-id'] == 'SL') & (data['break'] != 1)].copy()
filtered_data = filtered_data.rename(columns={
    "lf/hf": "lf_hf",
    "part-id": "part_id",
    "hr-clean": "hr_clean",
    "skin-temp-wrist-smooth": "skin_temp_wrist_smooth"
})
filtered_data['datetime'] = pd.to_datetime(filtered_data['datetime'])

downsampled_data = filtered_data.groupby(['part_id', 'scenario'], group_keys=False).apply(resample_within_groups)

for variable in downsample_variables:
    downsampled_data = filter_outliers_iqr(downsampled_data, variable)
for variable in original_resolution_variables:
    filtered_data = filter_outliers_iqr(filtered_data, variable)

# ------------------------ Modeling & exports ------------------------

results = {}
master_contrasts = []
contrasts_by_var = {}

processed_vars = []  # track which variables successfully wrote contrasts
warnings_log = []
scenarios = ['T1', 'T2', 'T3', 'T4']
pairs = [('T1','T2'),('T1','T3'),('T1','T4'),('T2','T3'),('T2','T4'),('T3','T4')]

for variable, settings in variables.items():
    if variable in downsampled_data.columns:
        data_for_model = downsampled_data
    elif variable in filtered_data.columns:
        data_for_model = filtered_data
    else:
        warnings_log.append(f"[WARN] Skipping {variable}: not found in downsampled_data or filtered_data columns.")
        continue

    if 'part_id' not in data_for_model.columns:
        warnings_log.append(f"[WARN] Skipping {variable}: 'part_id' column missing in selected dataset.")
        continue

    ylim = settings['ylim']
    spacing = settings['spacing']
    start_offset = settings['start_offset']

    # use the dataset that actually contains this variable
    data_for_model_use = data_for_model
    try:
        model = smf.mixedlm(
            f"{variable} ~ scenario",
            data_for_model_use,
            groups=data_for_model_use['part_id'],
            re_formula="~scenario"
        )
        result = model.fit()
    except Exception as e:
        warnings_log.append(f"[ERROR] Model failed for {variable}: {e}")
        continue
    results[variable] = result

    fe = result.fe_params
    bse = result.bse
    zval = fe / bse
    pvals = result.pvalues
    ci = result.conf_int()
    model_output = pd.DataFrame({
        "Coefficient": fe,
        "Std. Error": bse,
        "z-value": zval,
        "p": pvals,
        "CI Lower": ci[0],
        "CI Upper": ci[1]
    })
    model_output.to_csv(os.path.join(output_dir, f"00_Model_Output_{output_prefix}_{variable}.csv"), index=True)

    doc = Document()
    doc.add_heading(f'Mixed-Effects Model Output for {variable}', level=1)
    model_info = [
        f"Model: {result.model.formula}",
        f"No. Observations: {int(result.nobs)}",
        f"No. Groups: {len(result.model.group_labels)}",
        f"Log-Likelihood: {result.llf:.4f}",
        f"Scale (resid var): {result.scale:.4f}",
        f"Converged: {'Yes' if result.converged else 'No'}",
    ]
    doc.add_paragraph("Model Information:")
    for info in model_info:
        doc.add_paragraph(info)

    doc.add_paragraph("Coefficients:")
    coeff_table = doc.add_table(rows=1, cols=7)
    coeff_table.style = 'Light List'
    hdr_cells = coeff_table.rows[0].cells
    hdr_cells[0].text = 'Name'
    hdr_cells[1].text = 'Coefficient'
    hdr_cells[2].text = 'Std. Error'
    hdr_cells[3].text = 'z-value'
    hdr_cells[4].text = 'p'
    hdr_cells[5].text = 'CI Lower'
    hdr_cells[6].text = 'CI Upper'

    for name in fe.index:
        row_cells = coeff_table.add_row().cells
        row_cells[0].text = name
        row_cells[1].text = f"{fe[name]:.3f}"
        row_cells[2].text = f"{bse[name]:.3f}"
        row_cells[3].text = f"{zval[name]:.3f}"
        row_cells[4].text = f"{pvals[name]:.3e}"
        row_cells[5].text = f"{ci.loc[name, 0]:.3f}"
        row_cells[6].text = f"{ci.loc[name, 1]:.3f}"

    doc.save(os.path.join(output_dir, f"00_Model_Output_{output_prefix}_{variable}.docx"))

    residuals = result.resid
    plt.figure(figsize=(5, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of Residuals for {variable}")
    plt.grid(visible=True, linestyle="--", axis="both")
    plt.savefig(os.path.join(output_dir, f"{output_prefix}_{variable}_qq_plot.png"))
    plt.close()

    rows = []
    for A, B in pairs:
        est, se, z, p, d, lo, hi = contrast_from_fit(result, A, B)
        rows.append({
            'variable': variable,
            'contrast': f'{A} - {B}',
            'level_A': A,
            'level_B': B,
            'estimate': est,
            'std_error': se,
            'z_value': z,
            'p_raw': p,
            'cohens_d': d,
            'ci_low': lo,
            'ci_high': hi
        })
    df_contrasts = pd.DataFrame(rows)
    mask = df_contrasts['p_raw'].notna()
    qvals = np.full(len(df_contrasts), np.nan)
    if mask.any():
        qvals[mask] = multipletests(df_contrasts.loc[mask, 'p_raw'], method='fdr_bh')[1]
    df_contrasts['q'] = qvals
    df_contrasts['signif'] = np.where(df_contrasts['q'] < 0.05, 'T', '')

    df_contrasts.to_csv(os.path.join(output_dir, f"00_Contrasts_{output_prefix}_{variable}.csv"), index=False)
    contrasts_by_var[variable] = df_contrasts.copy()
    master_contrasts.append(df_contrasts)
    processed_vars.append(variable)

if master_contrasts:
    master = pd.concat(master_contrasts, ignore_index=True)
    master.to_csv(os.path.join(output_dir, "Contrasts_master.csv"), index=False)
    with pd.ExcelWriter(os.path.join(output_dir, "Contrasts_master.xlsx"), engine='xlsxwriter') as xw:
        for variable in processed_vars:
            dfv = master[master['variable'] == variable]
            if not dfv.empty:
                dfv.to_excel(xw, sheet_name=variable[:31], index=False)


# write a small log so it's obvious what was included
log_path = os.path.join(output_dir, "_export_log.txt")
with open(log_path, "w", encoding="utf-8") as fh:
    fh.write("Processed variables with contrasts: " + ", ".join(processed_vars) + "\n")
    if warnings_log:
        fh.write("\nWarnings/Errors:\n" + "\n".join(warnings_log) + "\n")


# Combined multi-panel plot (PNG + JPEG)
variables_to_plot = list(variables.keys())
titles = variable_titles


fig, axes = plt.subplots(1, len(variables_to_plot), figsize=(18 / 2.54, 7 / 2.54), sharey=False)

# Use numeric positions to make bracket drawing robust
x_positions = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
x_labels = ['T1', 'T2', 'T3', 'T4']

for i, variable in enumerate(variables_to_plot):
    ax = axes[i]
    res = results.get(variable)
    if res is None:
        ax.axis('off')
        continue

    fe = res.fe_params
    bse = res.bse

    # Compute scenario means and CIs from the model (your original logic)
    ci = {
        "T1": (fe['Intercept'] - 1.96 * bse['Intercept'], fe['Intercept'] + 1.96 * bse['Intercept'])
    }
    for sc in ['T2', 'T3', 'T4']:
        k = f"scenario[T.{sc}]"
        mu = fe['Intercept'] + fe[k]
        ci[sc] = (mu - 1.96 * bse[k], mu + 1.96 * bse[k])

    means = [fe['Intercept']] + [fe['Intercept'] + fe[f"scenario[T.{sc}]"] for sc in ['T2','T3','T4']]
    lower_bounds = [ci[s][0] for s in x_labels]
    upper_bounds = [ci[s][1] for s in x_labels]

    # Plot means with 95% CI as error bars
    errs = np.vstack([np.array(means) - np.array(lower_bounds),
                      np.array(upper_bounds) - np.array(means)])
    ax.errorbar(list(x_positions.values()), means,
                yerr=errs,
                fmt='o', markersize=1.5, color='black', ecolor='black',
                capsize=1, elinewidth=1, capthick=1)

    # Cosmetics and limits
    ax.set_title(titles[variable], fontsize=8, pad=10)
    ax.tick_params(axis='both', labelsize=7)
    ylim = variables[variable]['ylim']
    ax.set_ylim(ylim)
    step = gridline_steps[variable]
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(visible=True, linestyle='--', axis='y', linewidth=0.5, color='#2e2e2e')
    ax.set_xticks(list(x_positions.values()), labels=x_labels)

    # --- NEW: draw Cohen's d brackets above the pairs ---
    dfc = contrasts_by_var.get(variable)
    if dfc is not None and not dfc.empty:
        yrange = ylim[1] - ylim[0]
        # base height just above the tallest CI, plus any per-variable tuning
        top_ci = max(upper_bounds)
        start_offset = variables[variable]['start_offset']  # your per-variable tweak
        spacing = variables[variable]['spacing']            # your per-variable spacing
        y_base = top_ci + 0.06 * yrange + start_offset
        draw_cohens_d_brackets(
            ax=ax,
            df_contrasts=dfc,
            x_positions=x_positions,
            y_base=y_base,
            spacing=spacing,
            yrange=yrange
        )

axes[2].set_xlabel('Test scenarios', fontsize=7, labelpad=8, ha='center')
plt.tight_layout()

plt.savefig(os.path.join(output_dir, f"{output_prefix}_combined_variables_plot.png"), dpi=1200)
plt.savefig(os.path.join(output_dir, f"{output_prefix}_combined_variables_plot.jpeg"), dpi=1200)

plt.show()
plt.close()

