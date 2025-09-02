from pathlib import Path
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 00:00:32 2025
Christmas break, yay.

@author: kobas
"""
import pandas as pd
import numpy as np
import os, warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib import rcParams
from itertools import combinations
import statsmodels.formula.api as smf
from scipy.stats import zscore
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2


# Santas little helpers
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

# Define a function for Z-score-based filtering
def filter_outliers_zscore(df, column, threshold=3):
    """Z-score calculation within-subjects, within scenarios."""
    if column in df.columns:
        # Apply Z-score filtering within each group
        df = df.groupby(['part_id', 'scenario'], group_keys=False).apply(
            lambda group: group[(np.abs(zscore(group[column], nan_policy='omit')) <= threshold) | group[column].isna()]
        )
    return df


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
    # If random effects dimension >1 we likely had slopes (~C(sex)); else intercept-only
    try:
        return "~C(sex)" if res.cov_re.shape[0] > 1 else "~1"
    except Exception:
        return "~1"

def _fit_reduced_no_sex(varname, res_full, df):
    re_formula = _same_random_formula(res_full)
    # Reduced model: keep scenario, drop sex + scenario:sex
    formula_reduced = f"{varname} ~ scenario"
    try:
        red = smf.mixedlm(formula_reduced, data=df, groups=df["part_id"], re_formula=re_formula).fit()
    except Exception:
        # conservative fallback: random intercept
        red = smf.mixedlm(formula_reduced, data=df, groups=df["part_id"], re_formula="~1").fit()
    return red


def get_fe_cov(res) -> pd.DataFrame:
    """Get fixed effects only."""
    cov_all = res.cov_params()
    fe_idx = res.fe_params.index
    return cov_all.loc[fe_idx, fe_idx]

def cell_coef_series(res, scenario, sex) -> pd.Series:
    idx = res.fe_params.index
    v = pd.Series(0.0, index=idx)
    if 'Intercept' in idx:
        v['Intercept'] = 1.0
    if scenario in ['T2','T3','T4']:
        s = f"scenario[T.{scenario}]"
        if s in idx:
            v[s] = 1.0
    if sex == 'M':
        c = "C(sex, Treatment(reference='F'))[T.M]"
        if c in idx:
            v[c] = 1.0
        if scenario in ['T2','T3','T4']:
            i = f"scenario[T.{scenario}]:{c}"
            if i in idx:
                v[i] = 1.0
    return v

def cell_mean_ci(res, scenario, sex, z=1.96):
    beta  = res.fe_params
    covFE = get_fe_cov(res)
    v     = cell_coef_series(res, scenario, sex).reindex(covFE.index).fillna(0.0)
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


#%% 0. Set variables
# Working directory (change if needed)
# Quiet font warnings
warnings.filterwarnings("ignore", message=".*findfont:.*")

# ==== Repo-relative paths (no private absolute paths) ====
ROOT = Path(__file__).resolve().parents[1]  # repo root (assuming scripts live in 02_Codes/)
DATA   = ROOT / "01 Data"
GRAPHS = ROOT / "03 Graphs"
OUT    = ROOT / "04 Model Outputs"
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

# Sex levels + colors/markers/offsets
sexes           = ['F', 'M']
scenarios       = ["T1", "T2", "T3", "T4"]
sex_colors  = {'F': '#121417', 'M': '#FF1F62'}
sex_markers = {'F': 'o', 'M': 'o'}
x_shift         = {'F': 0.0, 'M': 0.1}

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
    'hr_clean':   {'ylim': (70, 94),   'spacing': 1.5,  'start_offset': -0.5},
    'rmssd':      {'ylim': (120, 220), 'spacing': 7,    'start_offset': -15},
    'lf_hf':      {'ylim': (0.3, 0.9), 'spacing': 0.04, 'start_offset': 0.02},
    'skin_temp':  {'ylim': (33.5, 37.5),   'spacing': 0.3,  'start_offset': 0},
    'EDA_Tonic':  {'ylim': (-2, 16),   'spacing': 1.4,  'start_offset': -2},
}

sex_display = {'F': 'Female', 'M': 'Male'}

# Significance thresholds for annotations
d_threshold = 0.44
p_threshold = 0.05

# Output folder (keep consistent across your pipeline)
out_dir = str((OUT / '02_Model_Outputs_Sex'))
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
data['sex']  = pd.Categorical(data['sex'],  categories=sexes,  ordered=True)

# Ensure participant IDs are categorical
data['part_id'] = data['part_id'].astype('category')

# Cleaned dataset for modeling
data_for_model = data.copy()


#%% 2. Fit models 
"""Model assumes participants as random effects with random intercepts, 
    but draws random slopes per Sex level. Random slopes per 
    Participant cause non-convergence."""

results, errors = {}, {}
for variable in variables:
    formula = f"{variable} ~ scenario * C(sex, Treatment(reference='F'))"
    try:
        # Try random intercept + slope for sex
        model = smf.mixedlm(formula, data=data_for_model,
                            groups=data_for_model['part_id'],
                            re_formula="~C(sex)") 
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
    for sex in sexes:
        for sA, sB in combinations(scenarios, 2):
            vA = cell_coef_series(res, sA, sex)
            vB = cell_coef_series(res, sB, sex)
            est, se, z, p, d, ci_lo, ci_hi = contrast_stats(res, vA, vB)
            rows.append({
                'variable': variable,
                'contrast_type': 'within_sex_across_scenarios',
                'sex': sex,
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
        vA = cell_coef_series(res, s, 'M')
        vB = cell_coef_series(res, s, 'F')
        est, se, z, p, d, ci_lo, ci_hi = contrast_stats(res, vA, vB)
        rows.append({
            'variable': variable,
            'contrast_type': 'within_scenario_across_sexes',
            'scenario': s,
            'level_A': 'M',
            'level_B': 'F',
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


#%% 3. Sex-focused omnibus table (Female vs Male block) --> separate Excel
sex_rows = []
for var, res_full in results.items():
    # Reduced model (no sex terms)
    res_red = _fit_reduced_no_sex(var, res_full, data_for_model)

    # LR test for the sex block
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

    sex_rows.append({
        "Variable": var,
        "LR stat (sex block)": lr_stat,
        "df": df_lrt,
        "p-value": p_lr,
        "Marginal R² (full)": r2m_full,
        "Conditional R² (full)": r2c_full,
        "ΔR²m due to sex": delta_r2m,
        "Cohen's f (sex)": f,
        "Residual var. (full)": resid_full,
        "Random var. (full)": rand_var,
    })

sex_effects_df = pd.DataFrame(sex_rows)

# Optional formatted copy
sex_fmt = sex_effects_df.copy()
for c in ["LR stat (sex block)", "Marginal R² (full)", "Conditional R² (full)",
          "ΔR²m due to sex", "Cohen's f (sex)", "Residual var. (full)", "Random var. (full)"]:
    sex_fmt[c] = sex_fmt[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
sex_fmt["p-value"] = sex_effects_df["p-value"].apply(lambda x: f"{x:.2E}" if pd.notna(x) else "")

# Write a separate Excel
excel_path_sex = os.path.join(out_dir, "MixedLM_SexBlock.xlsx")
with pd.ExcelWriter(excel_path_sex, engine="openpyxl") as w:
    sex_effects_df.to_excel(w, sheet_name="numeric", index=False)
    sex_fmt.to_excel(w, sheet_name="formatted", index=False)

print(f"Saved sex-focused table to: {excel_path_sex}")

    

#%% 4. Vis — model-based means/CI + sig annotations (above & below)
fig, axes = plt.subplots(1, len(variables), figsize=(18 / 2.54, 9 / 2.54), sharey=False)

# Build model-based cell estimates (means + 95% CI from fixed-effects covariance)
rows = []
for variable, res in results.items():
    for sc in scenarios:
        for cl in sexes:
            est, se, lo, hi = cell_mean_ci(res, sc, cl)
            rows.append({
                'variable': variable, 'scenario': sc, 'sex': cl,
                'estimate': est, 'std_error': se, 'ci_low': lo, 'ci_high': hi
            })
cell_estimates = pd.DataFrame(rows)

legend_handles, legend_labels = [], []
for i, variable in enumerate(variables):
    ax = axes[i]
    dfv = cell_estimates[cell_estimates['variable'] == variable].copy()

    x = np.arange(len(scenarios))
    # Plot means + CI for each sex (Female/Male)
    for cl in sexes:
        dcl = dfv[dfv['sex'] == cl].set_index('scenario').loc[scenarios]
        x_pos = [xi + x_shift[cl] for xi in x]
        h = ax.errorbar(
            x_pos,
            dcl['estimate'].values,
            yerr=[
                (dcl['estimate'] - dcl['ci_low']).values,
                (dcl['ci_high'] - dcl['estimate']).values
            ],
            fmt=sex_markers[cl],
            markersize=1.5,
            color=sex_colors[cl],
            ecolor=sex_colors[cl],
            capsize=1,
            elinewidth=1,
            capthick=1,
            label=sex_display[cl]
        )
        if i == 0:
            legend_handles.append(h)
            legend_labels.append(sex_display[cl])

    # Axes cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=7)
    ax.set_title(titles[variable], fontsize=8, pad=12)
    ax.tick_params(axis='both', pad=0.02, labelsize=7)
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
    fancybox=False, frameon=False, ncol=2, fontsize=7
)

# ---------- Annotations ----------
for i, variable in enumerate(variables):
    ax = axes[i]
    dfv = cell_estimates[cell_estimates['variable'] == variable]

    # --- Compute base y for top brackets ---
    max_upper = float(dfv['ci_high'].max())
    start_y   = max_upper + variables_all[variable]['start_offset']
    spacing   = variables_all[variable]['spacing']

    # -------- TOP: within-sex across-scenarios (bars above) --------
    top_df = contrasts_df[
        (contrasts_df['variable'] == variable) &
        (contrasts_df['contrast_type'] == 'within_sex_across_scenarios') &
        (contrasts_df['cohens_d'].abs() > d_threshold)
    ].copy()

    # map scenarios to x indices and order for tidy stacking
    top_df['x0'] = top_df['level_A'].map({s: j for j, s in enumerate(scenarios)})
    top_df['x1'] = top_df['level_B'].map({s: j for j, s in enumerate(scenarios)})
    top_df = top_df.sort_values(['sex', 'x0', 'x1'])

    # alpha by significance (Rule 1 vs Rule 2)
    top_df['alpha'] = np.where(top_df['p_value_fdr'] > p_threshold, 0.5, 1.0)


    for j, row in enumerate(top_df.itertuples(index=False)):
        x0, x1 = int(row.x0), int(row.x1)
        c      = row.sex
        y      = start_y + j * spacing
        color  = sex_colors[c]
        alpha  = float(row.alpha)

        # bracket line
        ax.plot([x0, x1], [y, y], color=color, linewidth=0.6, alpha=alpha)

        # label with white background box
        ax.text(
            (x0 + x1) / 2.0, y + spacing / 5.0,
            f"{row.cohens_d:.2f}",
            ha="center", va="bottom",
            fontsize=7, color=color, alpha=alpha,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.6)
        )

    # -------- BOTTOM: within-scenario across sexes (Male − Female) --------
    bot_df = contrasts_df[
        (contrasts_df['variable'] == variable) &
        (contrasts_df['contrast_type'] == 'within_scenario_across_sexes') &
        (contrasts_df['cohens_d'].abs() > d_threshold)
    ].copy()

    # alpha by significance
    bot_df['alpha'] = np.where(bot_df['p_value_fdr'] > p_threshold, 0.5, 1.0)

    # ensure we have space below x-axis labels
    ymin, ymax = ax.get_ylim()
    pad_bottom = 0.06 * (ymax - ymin)
    ax.set_ylim(ymin - pad_bottom, ymax)

    for row in bot_df.itertuples(index=False):
        sc_idx = scenarios.index(row.scenario)
        ax.text(
            sc_idx, ymin - pad_bottom * 2.2,
            f"{row.cohens_d:.2f}",
            ha="center", va="top",
            fontsize=7, color="black", alpha=float(row.alpha),
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.6)
        )


plt.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5)

# Save
output_prefix = os.path.join(out_dir, "Errorplots_by_Sex")
plt.savefig(f"{output_prefix}.pdf",  dpi=300)
plt.savefig(f"{output_prefix}.jpeg", dpi=300)
plt.show()
plt.close()

print(f"Saved model outputs to: {excel_path_models}")
print(f"Saved contrasts to: {excel_path_contrasts}")
