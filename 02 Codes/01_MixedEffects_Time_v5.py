from pathlib import Path
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 00:00:32 2025
Christmas break, yay.

@author: kobas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib import rcParams
import scipy.stats as stats
from scipy.stats import zscore
import matplotlib.ticker as ticker
from itertools import combinations
from scipy.stats import probplot
import os, warnings
from matplotlib.font_manager import FontProperties, fontManager
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

   

# Cohen's d calculation function
def calculate_cohens_d(group1, group2):
    """Cohen's d for paired (within-subjects) samples."""
    # Drop missing values from both groups at same indices
    paired_data = pd.DataFrame({'g1': group1, 'g2': group2}).dropna()
    differences = paired_data['g2'] - paired_data['g1']
    mean_diff = differences.mean()
    std_diff = differences.std(ddof=1)
    d = mean_diff / std_diff if std_diff != 0 else np.nan
    return d


def plot_qq_and_histogram(residuals, variable, output_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Q-Q Plot
    probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q Plot of Residuals for {variable}")
    axes[0].grid(visible=True, linestyle="--", axis="both")

    # Histogram
    axes[1].hist(residuals, bins=20, color='skyblue', edgecolor='black')
    axes[1].set_title(f"Histogram of Residuals for {variable}")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(visible=True, linestyle="--", axis="both")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_{variable}_qq_histogram.png")
    plt.show()
    plt.close()

# Define a function for Z-score-based filtering
def filter_outliers_zscore(df, column, threshold=3):
    if column in df.columns:
        # Apply Z-score filtering within each group
        df = df.groupby(['part_id', 'scenario'], group_keys=False).apply(
            lambda group: group[(np.abs(zscore(group[column], nan_policy='omit')) <= threshold) | group[column].isna()]
        )
    return df


def get_contrast_pval(result, contrast_vector):
    est = np.dot(contrast_vector, result.fe_params.values)
    se = np.sqrt(contrast_vector @ result.cov_params().values @ contrast_vector)
    t = est / se
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    return p

def compute_confidence_interval(fe_params, cov_matrix, coef_names):
    # Create full contrast vector for all model coefficients, default to 0
    contrast = np.zeros(len(fe_params))
    fe_index = list(fe_params.index)

    # Fill in contrast vector only where parameter exists
    for name in coef_names:
        if name in fe_index:
            idx = fe_index.index(name)
            contrast[idx] = 1.0

    # Compute standard error
    std_error = np.sqrt(contrast.T @ cov_matrix.values @ contrast)
    return std_error

def get_fe_cov(res) -> pd.DataFrame:
    cov_all = res.cov_params()
    fe_idx = res.fe_params.index
    return cov_all.loc[fe_idx, fe_idx]

def cell_coef_series(res, scenario, cluster) -> pd.Series:
    idx = res.fe_params.index
    v = pd.Series(0.0, index=idx)
    if 'Intercept' in idx:
        v['Intercept'] = 1.0
    if scenario in ['T2','T3','T4']:
        s = f"scenario[T.{scenario}]"
        if s in idx:
            v[s] = 1.0
    if cluster == 'Afternoon':
        c = "C(cluster, Treatment(reference='Morning'))[T.Afternoon]"
        if c in idx:
            v[c] = 1.0
        if scenario in ['T2','T3','T4']:
            i = f"scenario[T.{scenario}]:{c}"
            if i in idx:
                v[i] = 1.0
    return v

def cell_mean_ci(res, scenario, cluster, z=1.96):
    beta  = res.fe_params
    covFE = get_fe_cov(res)
    v     = cell_coef_series(res, scenario, cluster).reindex(covFE.index).fillna(0.0)
    est   = float(v @ beta.reindex(covFE.index))
    var   = float(v.values @ covFE.values @ v.values)
    se    = np.sqrt(var) if var >= 0 else np.nan
    lo    = est - z*se if np.isfinite(se) else np.nan
    hi    = est + z*se if np.isfinite(se) else np.nan
    return est, se, lo, hi

def contrast_stats(res, v1: pd.Series, v0: pd.Series):
    beta  = res.fe_params
    covFE = get_fe_cov(res)
    cvec = (v1 - v0).reindex(covFE.index).fillna(0.0)
    beta = beta.reindex(covFE.index)
    est = float(np.dot(cvec.values, beta.values))
    var = float(cvec.values @ covFE.values @ cvec.values)
    se  = np.sqrt(var) if var >= 0 else np.nan
    z = est / se if np.isfinite(se) and se > 0 else np.nan
    p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    d = est / np.sqrt(res.scale)
    ci_low  = est - 1.96 * se if np.isfinite(se) else np.nan
    ci_high = est + 1.96 * se if np.isfinite(se) else np.nan
    return est, se, z, p, d, ci_low, ci_high


#%% Set variables
# Working directory (change if needed)
# Quiet font warnings
warnings.filterwarnings("ignore", message=".*findfont:.*")

# Keep text editable in vector outputs
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype']  = 42

# ==== Repo-relative paths (no private absolute paths) ====
ROOT = Path(__file__).resolve().parents[1]  # repo root (assuming scripts live in 02_Codes/)
DATA   = ROOT / "01_Data"
GRAPHS = ROOT / "03_Graphs"
OUT    = ROOT / "04_Model_Outputs"
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



# Scenario colors (not used directly in this plot, but handy to keep)
scenario_colors = {'T1': '#22181C', 
                   'T2': '#59C3C3', 
                   'T3': '#FFAC27', 
                   'T4': '#EA002D'}

# Clusters + colors/markers/offsets
clusters        = ['Morning', 'Afternoon']
scenarios       = ["T1", "T2", "T3", "T4"]
cluster_colors  = {'Morning': '#121417', 'Afternoon': '#FF1F62'}
cluster_markers = {'Morning': 'o', 'Afternoon': 'o'}
x_shift         = {'Morning': 0.0, 'Afternoon': 0.1}

# Variables, y-limits and spacing used by the plot
variables = ['skin_temp', 'EDA_Tonic', 'hr_clean', 'rmssd', 'lf_hf']

titles = {
    'skin_temp': "Skin temperature (°C)",
    'EDA_Tonic': "EDA tonic (µS)",
    'hr_clean': "Heart rate (bpm)",
    'rmssd': "RMSSD (ms)",
    'lf_hf': "LF/HF ratio",
}

# Gridline spacing per panel
gridline_steps = {
    'skin_temp': 0.5,
    'EDA_Tonic': 2,
    'hr_clean': 2,
    'rmssd': 10,
    'lf_hf': 0.05,
}

# Panel limits and annotation offsets
variables_all = {
    'hr_clean':   {'ylim': (70, 92),   'spacing': 1.2,  'start_offset': 1},
    'rmssd':      {'ylim': (110, 220), 'spacing': 7,    'start_offset': 5},
    'lf_hf':      {'ylim': (0.3, 0.9), 'spacing': 0.04, 'start_offset': -0.06},
    'skin_temp':  {'ylim': (33, 37),   'spacing': 0.3,  'start_offset': 0.05},
    'EDA_Tonic':  {'ylim': (-2, 16),   'spacing': 1.3,  'start_offset': -0.5},
}

# Significance thresholds for annotations
d_threshold = 0.44
p_threshold = 0.05

# Output folder (keep consistent across your pipeline)
out_dir = str((OUT / '03_Model_Outputs_Circadian'))
os.makedirs(str(out_dir), exist_ok=True)

#%% 1. Import & Filter Data
data_full = pd.read_csv(str(DATA / '00_SL2201_CombinedData.csv'))

# Keep only relevant experiment runs
data = data_full[
    (data_full['lab-id'] == 'SL') &    # experiment label
    (data_full['break'] != 1)          # exclude break periods
].copy()

# Rename columns for consistent variable names
data = data.rename(columns={
    "time-elapsed": "time_elapsed",
    "lf/hf": "lf_hf",
    "part-id": "part_id",
    "hr-clean": "hr_clean",
    "skin-temp-wrist-smooth": "skin_temp"
})

# Filter out first & last 2 minutes
data = data[(data['time_elapsed'] >= 2) & (data['time_elapsed'] < 179)]

# Outlier removal using z-score (per participant × scenario)
for variable in variables:
    before = len(data)
    data = filter_outliers_zscore(data, variable, threshold=3)
    after = len(data)
    print(f"Rows after Z-score filtering for {variable}: {after} (removed {before - after})")

# Compute 10-minute bins
data['time_bin'] = (data['time_elapsed'] // 10) * 10

# Remove rows with missing values in critical variables
data = data.dropna(subset=['part_id', 'skin_temp', 'time_bin', 'hr_clean', 'lf_hf', 'rmssd', 'EDA_Tonic'])

# Ensure ordered categoricals with desired references
data['scenario'] = pd.Categorical(data['scenario'], categories=scenarios, ordered=True)
data['cluster']  = pd.Categorical(data['cluster'],  categories=clusters,  ordered=True)

# Ensure participant IDs are categorical
data['part_id'] = data['part_id'].astype('category')

# Cleaned dataset for modeling
data_for_model = data.copy()


#%% Fit models 
"""Model assumes participants as random effects with random intercepts, 
    but draws random slopes per Cluster level. Random slopes per 
    Participant cause non-convergence."""

results, errors = {}, {}
for variable in variables:
    formula = f"{variable} ~ scenario * C(cluster, Treatment(reference='Morning'))"
    try:
        # Try random intercept + slope for cluster
        model = smf.mixedlm(formula, data=data_for_model,
                            groups=data_for_model['part_id'],
                            re_formula="~C(cluster)") 
        res = model.fit()
    except Exception:
        # Fallback to random intercept only
        model = smf.mixedlm(formula, data=data_for_model,
                            groups=data_for_model['part_id'],
                            re_formula="~1")
        res = model.fit()
    results[variable] = res


# Export model outputs
coef_frames, fit_stats_rows, varcomp_rows = [], [], []
for var, res in results.items():
    fe_idx = res.fe_params.index
    fe_params = res.fe_params.loc[fe_idx]
    fe_bse    = res.bse.loc[fe_idx] if isinstance(res.bse, pd.Series) else pd.Series(res.bse, index=res.params.index).loc[fe_idx]
    fe_tvals  = res.tvalues.loc[fe_idx] if isinstance(res.tvalues, pd.Series) else pd.Series(res.tvalues, index=res.params.index).loc[fe_idx]
    fe_pvals  = res.pvalues.loc[fe_idx] if isinstance(res.pvalues, pd.Series) else pd.Series(res.pvalues, index=res.params.index).loc[fe_idx]
    ci_all = res.conf_int()
    ci_fe  = ci_all.loc[fe_idx]
    coef_df = pd.DataFrame({
        'variable': var,
        'term': fe_idx,
        'estimate': fe_params.values,
        'std_error': fe_bse.values,
        'z_value': fe_tvals.values,
        'p_value': fe_pvals.values,
        'ci_low': ci_fe.iloc[:, 0].values,
        'ci_high': ci_fe.iloc[:, 1].values
    })
    coef_frames.append(coef_df)
    fit_stats_rows.append({
        'variable': var,
        'nobs': int(res.nobs),
        'log_likelihood': float(res.llf),
        'aic': float(res.aic),
        'bic': float(res.bic),
        'scale_resid_variance': float(res.scale),
        'converged': bool(getattr(res, 'converged', True))
    })
    try:
        rand_var = float(res.cov_re.iloc[0, 0])
    except Exception:
        rand_var = np.nan
    varcomp_rows.append({'variable': var, 'component': 'random_intercept_variance', 'value': rand_var})
    varcomp_rows.append({'variable': var, 'component': 'residual_variance', 'value': float(res.scale)})

coef_all = pd.concat(coef_frames, ignore_index=True)
fit_stats = pd.DataFrame(fit_stats_rows)
varcomp   = pd.DataFrame(varcomp_rows)

excel_path_models = os.path.join(out_dir, "MixedLM_ModelOutputs.xlsx")
with pd.ExcelWriter(excel_path_models, engine="openpyxl") as writer:
    coef_all.to_excel(writer, sheet_name="coefficients", index=False)
    fit_stats.to_excel(writer, sheet_name="fit_stats", index=False)
    varcomp.to_excel(writer, sheet_name="variance_components", index=False)


# Contrasts
rows = []
for variable, res in results.items():
    for cluster in clusters:
        for sA, sB in combinations(scenarios, 2):
            vA = cell_coef_series(res, sA, cluster)
            vB = cell_coef_series(res, sB, cluster)
            est, se, z, p, d, ci_lo, ci_hi = contrast_stats(res, vA, vB)
            rows.append({
                'variable': variable,
                'contrast_type': 'within_cluster_across_scenarios',
                'cluster': cluster,
                'level_A': sA,
                'level_B': sB,
                'estimate_A_minus_B': est,
                'std_error': se,
                'z_value': z,
                'p_value_raw': p,
                'cohens_d': d,
                'ci_low': ci_lo,
                'ci_high': ci_hi
            })
    for s in scenarios:
        vA = cell_coef_series(res, s, 'Afternoon')
        vB = cell_coef_series(res, s, 'Morning')
        est, se, z, p, d, ci_lo, ci_hi = contrast_stats(res, vA, vB)
        rows.append({
            'variable': variable,
            'contrast_type': 'within_scenario_across_clusters',
            'scenario': s,
            'level_A': 'Afternoon',
            'level_B': 'Morning',
            'estimate_A_minus_B': est,
            'std_error': se,
            'z_value': z,
            'p_value_raw': p,
            'cohens_d': d,
            'ci_low': ci_lo,
            'ci_high': ci_hi
        })

contrasts_df = pd.DataFrame(rows)

mask = contrasts_df['p_value_raw'].notna()
_, p_adj, _, _ = multipletests(contrasts_df.loc[mask, 'p_value_raw'], method='fdr_bh')
contrasts_df.loc[mask, 'p_value_fdr'] = p_adj
contrasts_df['sig_plot'] = (contrasts_df['p_value_fdr'] < 0.05) & (contrasts_df['cohens_d'].abs() > 0.45)

excel_path_contrasts = os.path.join(out_dir, "MixedLM_Contrasts.xlsx")
with pd.ExcelWriter(excel_path_contrasts, engine="openpyxl") as writer:
    contrasts_df.to_excel(writer, sheet_name="contrasts", index=False)


#%% Cluster-focused omnibus table (Morning vs Afternoon block) --> separate Excel
from scipy.stats import chi2

def _predict_fixed(res):
    return res.model.exog @ res.fe_params

def _predict_random_contrib(res):
    try:
        Z = res.model.exog_re
        groups = np.asarray(res.model.groups)
        re_dict = res.random_effects
        out = np.zeros(len(groups), dtype=float)
        for i, (g, zrow) in enumerate(zip(groups, Z)):
            b = re_dict.get(g, re_dict.get(str(g), None))
            if b is None:
                continue
            out[i] = float(np.asarray(zrow) @ np.asarray(b))
        return out
    except Exception:
        return np.zeros(int(res.nobs), dtype=float)

def _r2_marginal_conditional(res):
    mu_fe  = _predict_fixed(res)
    var_fe = float(np.var(mu_fe, ddof=1))
    re_obs = _predict_random_contrib(res)
    var_re = float(np.var(re_obs, ddof=1))
    resid  = float(res.scale)
    denom  = var_fe + var_re + resid
    r2_m   = (var_fe / denom) if denom > 0 else np.nan
    r2_c   = ((var_fe + var_re) / denom) if denom > 0 else np.nan
    return r2_m, r2_c, var_re, resid

def _same_random_formula(res):
    # If random effects dimension >1 we likely had slopes (~C(cluster)); else intercept-only
    try:
        return "~C(cluster)" if res.cov_re.shape[0] > 1 else "~1"
    except Exception:
        return "~1"

def _fit_reduced_no_cluster(varname, res_full, df):
    re_formula = _same_random_formula(res_full)
    # Reduced model: keep scenario, drop cluster + scenario:cluster
    formula_reduced = f"{varname} ~ scenario"
    try:
        red = smf.mixedlm(formula_reduced, data=df, groups=df["part_id"], re_formula=re_formula).fit()
    except Exception:
        # conservative fallback: random intercept
        red = smf.mixedlm(formula_reduced, data=df, groups=df["part_id"], re_formula="~1").fit()
    return red

cluster_rows = []
for var, res_full in results.items():
    # Reduced model (no cluster terms)
    res_red = _fit_reduced_no_cluster(var, res_full, data_for_model)

    # LR test for the cluster block
    lr_stat = 2.0 * (res_full.llf - res_red.llf)
    df_lrt  = max(0, len(res_full.fe_params) - len(res_red.fe_params))  # usually 4
    p_lr    = chi2.sf(lr_stat, df_lrt) if df_lrt > 0 else np.nan

    # R²m / R²c for full and reduced; partial R²m for the block
    r2m_full, r2c_full, var_re_full, resid_full = _r2_marginal_conditional(res_full)
    r2m_red,  r2c_red,  _,             _        = _r2_marginal_conditional(res_red)
    delta_r2m = (r2m_full - r2m_red) if (pd.notna(r2m_full) and pd.notna(r2m_red)) else np.nan

    # Cohen's f for the added block (using marginal R²)
    f2 = (delta_r2m / (1 - r2m_full)) if (pd.notna(delta_r2m) and pd.notna(r2m_full) and r2m_full < 1) else np.nan
    f  = np.sqrt(f2) if pd.notna(f2) and f2 >= 0 else np.nan

    # Random variance summary (sum of RE variances). If you prefer intercept-only, use cov_re.iloc[0,0].
    try:
        rand_var = float(np.sum(np.diag(np.asarray(res_full.cov_re))))
    except Exception:
        rand_var = np.nan

    cluster_rows.append({
        "Variable": var,
        "LR stat (cluster block)": lr_stat,
        "df": df_lrt,
        "p-value": p_lr,
        "Marginal R² (full)": r2m_full,
        "Conditional R² (full)": r2c_full,
        "ΔR²m due to cluster": delta_r2m,
        "Cohen's f (cluster)": f,
        "Residual var. (full)": resid_full,
        "Random var. (full)": rand_var,
    })

cluster_effects_df = pd.DataFrame(cluster_rows)

# Optional formatted copy
cluster_fmt = cluster_effects_df.copy()
for c in ["LR stat (cluster block)", "Marginal R² (full)", "Conditional R² (full)",
          "ΔR²m due to cluster", "Cohen's f (cluster)", "Residual var. (full)", "Random var. (full)"]:
    cluster_fmt[c] = cluster_fmt[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
cluster_fmt["p-value"] = cluster_effects_df["p-value"].apply(lambda x: f"{x:.2E}" if pd.notna(x) else "")

# Write a separate Excel
excel_path_cluster = os.path.join(out_dir, "MixedLM_ClusterBlock.xlsx")
with pd.ExcelWriter(excel_path_cluster, engine="openpyxl") as w:
    cluster_effects_df.to_excel(w, sheet_name="numeric", index=False)
    cluster_fmt.to_excel(w, sheet_name="formatted", index=False)

print(f"Saved cluster-focused table to: {excel_path_cluster}")

    

#%% Vis — model-based means/CI + sig annotations (above & below)

fig, axes = plt.subplots(1, len(variables), figsize=(18 / 2.54, 8 / 2.54), sharey=False)

# Build model-based cell estimates (means + 95% CI from fixed-effects covariance)
rows = []
for variable, res in results.items():
    for sc in scenarios:
        for cl in clusters:
            est, se, lo, hi = cell_mean_ci(res, sc, cl)
            rows.append({
                'variable': variable, 'scenario': sc, 'cluster': cl,
                'estimate': est, 'std_error': se, 'ci_low': lo, 'ci_high': hi
            })
cell_estimates = pd.DataFrame(rows)

legend_handles, legend_labels = [], []
for i, variable in enumerate(variables):
    ax = axes[i]
    dfv = cell_estimates[cell_estimates['variable'] == variable].copy()

    x = np.arange(len(scenarios))
    # Plot means + CI for each cluster (Morning/Afternoon)
    for cl in clusters:
        dcl = dfv[dfv['cluster'] == cl].set_index('scenario').loc[scenarios]
        x_pos = [xi + x_shift[cl] for xi in x]
        h = ax.errorbar(
            x_pos,
            dcl['estimate'].values,
            yerr=[
                (dcl['estimate'] - dcl['ci_low']).values,
                (dcl['ci_high'] - dcl['estimate']).values
            ],
            fmt=cluster_markers[cl],
            markersize=1.5,
            color=cluster_colors[cl],
            ecolor=cluster_colors[cl],
            capsize=1,
            elinewidth=1,
            capthick=1,
            label=cl
        )
        if i == 0:
            legend_handles.append(h)
            legend_labels.append(cl)

    # Axes cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=6)
    ax.set_title(titles[variable], fontsize=7, pad=9)
    ax.tick_params(axis='both', pad=0.01, labelsize=6)
    ax.set_ylim(variables_all[variable]['ylim'])
    step = gridline_steps[variable]
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    ax.spines['left'].set_linewidth(1);   ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(1); ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
    ax.grid(visible=True, linestyle="--", axis="y", linewidth=0.5, color="#75726C", alpha=0.5)

# Shared x label
# axes[2].set_xlabel("Test scenarios", fontsize=7, labelpad=4, ha='center')

# Legend below all subplots
fig.legend(
    legend_handles, legend_labels,
    loc='lower center', bbox_to_anchor=(0.5, 0.01),
    fancybox=False, frameon=False, ncol=2, fontsize=6
)

# ---------- Annotations ----------
for i, variable in enumerate(variables):
    ax = axes[i]
    dfv = cell_estimates[cell_estimates['variable'] == variable]

    # Top annotations: within-cluster across-scenarios (bars above)
    max_upper = float(dfv['ci_high'].max())
    start_y   = max_upper + variables_all[variable]['start_offset']
    spacing   = variables_all[variable]['spacing']

    sig_top = contrasts_df[
        (contrasts_df['variable'] == variable) &
        (contrasts_df['contrast_type'] == 'within_cluster_across_scenarios') &
        (contrasts_df['p_value_fdr'] < (p_threshold if 'p_threshold' in globals() else 0.05)) &
        (contrasts_df['cohens_d'].abs() > (d_threshold if 'd_threshold' in globals() else 0.45))
    ].copy()

    sig_top['x0'] = sig_top['level_A'].map({s: j for j, s in enumerate(scenarios)})
    sig_top['x1'] = sig_top['level_B'].map({s: j for j, s in enumerate(scenarios)})
    sig_top = sig_top.sort_values(['cluster', 'x0', 'x1'])

    for j, row in enumerate(sig_top.itertuples(index=False)):
        x0, x1 = int(row.x0), int(row.x1)
        c = row.cluster
        y = start_y + j * spacing
        ax.plot([x0, x1], [y, y], color=cluster_colors[c], linewidth=0.5)
        ax.text((x0 + x1)/2, y + spacing/5, f"{row.cohens_d:.2f}",
                ha="center", fontsize=5, color=cluster_colors[c])

    # Bottom annotations: within-scenario across-clusters (Afternoon − Morning) under tick
    sig_bot = contrasts_df[
        (contrasts_df['variable'] == variable) &
        (contrasts_df['contrast_type'] == 'within_scenario_across_clusters') &
        (contrasts_df['p_value_fdr'] < p_threshold) &
        (contrasts_df['cohens_d'].abs() > d_threshold)
    ].copy()
    
    # Add more bottom padding so text is below x-axis labels
    ymin, ymax = ax.get_ylim()
    pad_bottom = 0.06 * (ymax - ymin)
    ax.set_ylim(ymin - pad_bottom, ymax)
    
    for row in sig_bot.itertuples(index=False):
        sc_idx = scenarios.index(row.scenario)
        # Place text well below tick labels
        ax.text(
            sc_idx, ymin - pad_bottom * 2.2,
            f"{row.cohens_d:.2f}",
            ha="center", va="top",
            fontsize=5, color="black"  
        )

plt.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5)

# Save
output_prefix = os.path.join(out_dir, "Errorplots_by_TimeofDay")
plt.savefig(f"{output_prefix}.pdf",  dpi=300)
plt.savefig(f"{output_prefix}.jpeg", dpi=300)
plt.show()
plt.close()

print(f"Saved model outputs to: {excel_path_models}")
print(f"Saved contrasts to: {excel_path_contrasts}")
