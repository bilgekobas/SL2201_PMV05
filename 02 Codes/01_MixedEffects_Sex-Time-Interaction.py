# -*- coding: utf-8 -*-
"""
Sex × Time-of-day mixed-model plots (styling preserved)
- One MixedLM/variable with scenario * cluster * sex (fixed), participant random
- Cell means + 95% CI from fixed-effect covariance
- Model-based contrasts with FDR and Cohen's d

Outputs (in ./04_SexTime_Outputs/):
  - MixedLM_SexTime_ModelOutputs.xlsx
  - MixedLM_SexTime_Contrasts.xlsx
  - ErrorPlots_Cluster_Comparison.(pdf/jpeg)   # rows=Sex, overlay Morning/Afternoon
  - ErrorPlots_Sex_Comparison.(pdf/jpeg)       # rows=Cluster, overlay Female/Male
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib import rcParams
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore, norm
from itertools import combinations
from scipy.stats import chi2  


# ---------------- Helpers ----------------

def filter_outliers_zscore(df, column, threshold=3):
    """Z-score filter within participant × scenario (keeps NaNs)."""
    if column not in df.columns:
        return df
    return (
        df.groupby(['part_id', 'scenario'], group_keys=False)
          .apply(lambda g: g[(np.abs(zscore(g[column], nan_policy='omit')) <= threshold) | g[column].isna()])
    )

def _fe_cov(res) -> pd.DataFrame:
    cov = res.cov_params()
    idx = res.fe_params.index
    return cov.loc[idx, idx]

def _cell_vector_3way(res, scenario, cluster, sex) -> pd.Series:
    """Build FE contrast vector for cell mean under treatment coding."""
    idx  = res.fe_params.index
    v    = pd.Series(0.0, index=idx)

    cname = "C(cluster, Treatment(reference='Morning'))[T.Afternoon]"
    sname = "C(sex, Treatment(reference='F'))[T.M]"

    # Intercept
    if 'Intercept' in idx:
        v['Intercept'] = 1.0

    # Main effects
    if scenario in ['T2', 'T3', 'T4']:
        s_term = f"scenario[T.{scenario}]"
        if s_term in idx: v[s_term] = 1.0
    if cluster == 'Afternoon' and cname in idx:
        v[cname] = 1.0
    if sex == 'M' and sname in idx:
        v[sname] = 1.0

    # Two-way
    if scenario in ['T2','T3','T4'] and cluster == 'Afternoon':
        t = f"scenario[T.{scenario}]:{cname}"
        if t in idx: v[t] = 1.0
    if scenario in ['T2','T3','T4'] and sex == 'M':
        t = f"scenario[T.{scenario}]:{sname}"
        if t in idx: v[t] = 1.0
    if cluster == 'Afternoon' and sex == 'M':
        t = f"{cname}:{sname}"
        if t in idx: v[t] = 1.0

    # Three-way
    if scenario in ['T2','T3','T4'] and cluster == 'Afternoon' and sex == 'M':
        t = f"scenario[T.{scenario}]:{cname}:{sname}"
        if t in idx: v[t] = 1.0

    return v

def cell_mean_ci(res, scenario, cluster, sex, z=1.96):
    beta = res.fe_params
    cov  = _fe_cov(res)
    v    = _cell_vector_3way(res, scenario, cluster, sex).reindex(cov.index).fillna(0.0)
    est  = float(v @ beta.reindex(cov.index))
    var  = float(v.values @ cov.values @ v.values)
    se   = np.sqrt(var) if var >= 0 else np.nan
    lo   = est - z*se if np.isfinite(se) else np.nan
    hi   = est + z*se if np.isfinite(se) else np.nan
    return est, se, lo, hi

def contrast_stats(res, v1: pd.Series, v0: pd.Series):
    cov  = _fe_cov(res)
    beta = res.fe_params.reindex(cov.index)
    cvec = (v1 - v0).reindex(cov.index).fillna(0.0)

    est = float(cvec.values @ beta.values)
    var = float(cvec.values @ cov.values @ cvec.values)
    se  = np.sqrt(var) if var >= 0 else np.nan
    z   = est / se if (np.isfinite(se) and se > 0) else np.nan
    p   = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    d   = est / np.sqrt(res.scale)
    lo  = est - 1.96*se if np.isfinite(se) else np.nan
    hi  = est + 1.96*se if np.isfinite(se) else np.nan
    return est, se, z, p, d, lo, hi


def _draw_sig_bars(ax, rows, color, x_shift, start_y, spacing):
    """Draw horizontal sig bars stacked from start_y; return next y."""
    cur_y = start_y
    for _, r in rows.iterrows():
        x0, x1 = int(r['x0']), int(r['x1'])
        ax.hlines(y=cur_y, xmin=x0 + x_shift, xmax=x1 + x_shift, color=color, linewidth=1)
        ax.text((x0 + x1)/2 + x_shift, cur_y + spacing/5, f"{r['cohens_d']:.2f}",
                ha="center", fontsize=12, color=color)
        cur_y += spacing
    return cur_y


def _predict_fixed(res):
    return res.model.exog @ res.fe_params

def _predict_random_contrib(res):
    """Per-row random contribution Z b. Falls back to zeros if unavailable."""
    try:
        Z = res.model.exog_re
        groups = np.asarray(res.model.groups)
        re = res.random_effects
        out = np.zeros(len(groups), dtype=float)
        for i, (g, zrow) in enumerate(zip(groups, Z)):
            b = re.get(g, re.get(str(g), None))
            if b is None:
                continue
            out[i] = float(np.asarray(zrow) @ np.asarray(b))
        return out
    except Exception:
        return np.zeros(int(res.nobs), dtype=float)

def _r2_marginal_conditional(res):
    """Nakagawa-style R² using per-row FE/RE predictions from the fitted model."""
    mu_fe  = _predict_fixed(res)
    var_fe = float(np.var(mu_fe, ddof=1))
    re_obs = _predict_random_contrib(res)
    var_re = float(np.var(re_obs, ddof=1))
    resid  = float(res.scale)
    denom  = var_fe + var_re + resid
    r2m    = (var_fe / denom) if denom > 0 else np.nan
    r2c    = ((var_fe + var_re) / denom) if denom > 0 else np.nan
    return r2m, r2c, var_re, resid

def _same_random_formula(res):
    """Mirror the random structure you ended up with (slopes vs intercept)."""
    try:
        return "~C(cluster)" if res.cov_re.shape[0] > 1 else "~1"
    except Exception:
        return "~1"

def _fit_ml(formula, df, re_formula):
    """Fit MixedLM with ML, keep the requested random structure; fallback to ~1 if needed."""
    try:
        m = smf.mixedlm(formula, data=df, groups=df["part_id"], re_formula=re_formula)
        return m.fit(reml=False)
    except Exception:
        m = smf.mixedlm(formula, data=df, groups=df["part_id"], re_formula="~1")
        return m.fit(reml=False)



#%% 0. Config (styling preserved) 

# Working dir
os.chdir(r"C:\Users\kobas\OneDrive\Desktop\Meta\SL2201 PMV")
print("Current working directory:", os.getcwd())

warnings.filterwarnings("ignore", message=".*findfont:.*")

# Custom font (keeps your look)
try:
    custom_font_path = r".\Fonts\Archivo_SemiCondensed-Medium.ttf"
    fontManager.addfont(custom_font_path)
    custom_font = FontProperties(fname=custom_font_path)
    rcParams['font.family'] = custom_font.get_name()
except Exception:
    pass

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype']  = 42

# Factors/vars
sexes     = ['F', 'M']
clusters  = ['Morning', 'Afternoon']
scenarios = ['T1', 'T2', 'T3', 'T4']
variables = ["skin_temp", "EDA_Tonic", "hr_clean", "rmssd", "lf_hf"]

variable_settings = {
    'skin_temp': {'ylim': (33.5, 37.5),  'spacing': 0.25, 'manual_offset': -0.75, 'gridline_step': 0.5, 'title': "Skin Temperature (°C)"},
    'EDA_Tonic': {'ylim': (-3, 14),    'spacing': 0.85, 'manual_offset': -4.0, 'gridline_step': 1.0, 'title': "EDA Tonic (µS)"},
    'hr_clean':  {'ylim': (65, 100),   'spacing': 2.00, 'manual_offset': -5.0, 'gridline_step': 2.0, 'title': "Heart Rate (bpm)"},
    'rmssd':     {'ylim': (90, 250),   'spacing': 8.00, 'manual_offset': -15.0, 'gridline_step': 10.0,'title': "RMSSD (ms)"},
    'lf_hf':     {'ylim': (0.3, 1.0),  'spacing': 0.05, 'manual_offset': -0.05, 'gridline_step': 0.05, 'title': "LF/HF Ratio"},
}

# Colors/markers/offsets (your palette)
cluster_colors  = {'Morning': '#121417', 'Afternoon': '#FF1F62'}
cluster_markers = {'Morning': 'o',       'Afternoon': 'o'}
cluster_shift   = {'Morning': -0.1,      'Afternoon': 0.1}

sex_colors  = {'F': '#121417', 'M': '#FF1F62'}
sex_markers = {'F': 'o',       'M': 'o'}
sex_shift   = {'F': -0.1,      'M': 0.1}

# Anno thresholds
P_THRESH = 0.05
D_THRESH = 0.45

# Out dir
OUT_DIR = "04_SexTime_Outputs"
os.makedirs(OUT_DIR, exist_ok=True)

#%% Load & filter 

data_full = pd.read_csv(r".\Data\00_SL2201_CombinedData.csv")
data = (
    data_full[(data_full['lab-id'] == 'SL') & (data_full['break'] != 1)]
      .rename(columns={
          "time-elapsed": "time_elapsed",
          "lf/hf": "lf_hf",
          "part-id": "part_id",
          "hr-clean": "hr_clean",
          "skin-temp-wrist-smooth": "skin_temp"
      })
)

# Trim first/last ~2 min
data = data[(data['time_elapsed'] >= 2) & (data['time_elapsed'] < 179)]

# Outlier filtering
for v in variables:
    before = len(data)
    data = filter_outliers_zscore(data, v, threshold=3)
    after = len(data)
    print(f"Rows after Z-score filtering for {v}: {after} (removed {before - after})")

# Keep complete cases
data = data.dropna(subset=['part_id','scenario','cluster','sex','skin_temp','EDA_Tonic','hr_clean','rmssd','lf_hf'])

# Categories (ordered; references: Morning, F, T1 via treatment coding)
data['scenario'] = pd.Categorical(data['scenario'], categories=scenarios, ordered=True)
data['cluster']  = pd.Categorical(data['cluster'],  categories=clusters,  ordered=True)
data['sex']      = pd.Categorical(data['sex'],      categories=sexes,     ordered=True)
data['part_id']  = data['part_id'].astype('category')


#%% Fit models 

results, errors = {}, {}
for v in variables:
    formula = f"{v} ~ scenario * C(cluster, Treatment(reference='Morning')) * C(sex, Treatment(reference='F'))"
    try:
        # Try random intercept + cluster slope; fall back to intercept-only
        try:
            model = smf.mixedlm(formula, data=data, groups=data['part_id'], re_formula="~C(cluster)")
            res = model.fit()
        except Exception:
            model = smf.mixedlm(formula, data=data, groups=data['part_id'], re_formula="~1")
            res = model.fit()
        results[v] = res
    except Exception as e:
        errors[v] = str(e)

if errors:
    print("\n[WARN] Model issues:")
    for k, msg in errors.items():
        print(f"  - {k}: {msg}")

#%% Excel: model outputs 

coef_frames, fit_stats_rows, varcomp_rows = [], [], []
for var, res in results.items():
    fe_idx   = res.fe_params.index
    fe_bse   = (res.bse if isinstance(res.bse, pd.Series) else pd.Series(res.bse, index=res.params.index)).loc[fe_idx]
    fe_tvals = (res.tvalues if isinstance(res.tvalues, pd.Series) else pd.Series(res.tvalues, index=res.params.index)).loc[fe_idx]
    fe_pvals = (res.pvalues if isinstance(res.pvalues, pd.Series) else pd.Series(res.pvalues, index=res.params.index)).loc[fe_idx]
    ci_fe    = res.conf_int().loc[fe_idx]

    coef_frames.append(pd.DataFrame({
        'variable': var,
        'term': fe_idx,
        'estimate': res.fe_params.loc[fe_idx].values,
        'std_error': fe_bse.values,
        'z_value': fe_tvals.values,
        'p_value': fe_pvals.values,
        'ci_low': ci_fe.iloc[:,0].values,
        'ci_high': ci_fe.iloc[:,1].values
    }))

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
        rand_var = float(res.cov_re.iloc[0,0])
    except Exception:
        rand_var = np.nan
    varcomp_rows += [
        {'variable': var, 'component': 'random_intercept_variance', 'value': rand_var},
        {'variable': var, 'component': 'residual_variance',         'value': float(res.scale)}
    ]

coef_all = pd.concat(coef_frames, ignore_index=True)
fit_stats = pd.DataFrame(fit_stats_rows)
varcomp   = pd.DataFrame(varcomp_rows)

excel_models = os.path.join(OUT_DIR, "MixedLM_SexTime_ModelOutputs.xlsx")
with pd.ExcelWriter(excel_models, engine="openpyxl") as w:
    coef_all.to_excel(w, sheet_name="coefficients", index=False)
    fit_stats.to_excel(w, sheet_name="fit_stats", index=False)
    varcomp.to_excel(w, sheet_name="variance_components", index=False)
print(f"[OK] Wrote: {excel_models}")

#%% Contrasts (FDR) 

rows = []
for var, res in results.items():
    # A) Within-sex, within-cluster across scenarios (top bars)
    for sex in sexes:
        for cl in clusters:
            for sA, sB in combinations(scenarios, 2):
                vA = _cell_vector_3way(res, sA, cl, sex)
                vB = _cell_vector_3way(res, sB, cl, sex)
                est, se, z, p, d, lo, hi = contrast_stats(res, vA, vB)
                rows.append({
                    'variable': var, 'contrast_type': 'within_sex_within_cluster_across_scenarios',
                    'sex': sex, 'cluster': cl, 'level_A': sA, 'level_B': sB,
                    'estimate_A_minus_B': est, 'std_error': se, 'z_value': z,
                    'p_value_raw': p, 'cohens_d': d, 'ci_low': lo, 'ci_high': hi
                })

    # B) Within-sex, within-scenario: Afternoon − Morning (bottom labels in Fig A)
    for sex in sexes:
        for s in scenarios:
            vA = _cell_vector_3way(res, s, 'Afternoon', sex)
            vB = _cell_vector_3way(res, s, 'Morning',   sex)
            est, se, z, p, d, lo, hi = contrast_stats(res, vA, vB)
            rows.append({
                'variable': var, 'contrast_type': 'within_sex_within_scenario_Afternoon_minus_Morning',
                'sex': sex, 'scenario': s, 'level_A': 'Afternoon', 'level_B': 'Morning',
                'estimate_A_minus_B': est, 'std_error': se, 'z_value': z,
                'p_value_raw': p, 'cohens_d': d, 'ci_low': lo, 'ci_high': hi
            })

    # C) Within-cluster, within-scenario: Male − Female (bottom labels in Fig B)
    for cl in clusters:
        for s in scenarios:
            vA = _cell_vector_3way(res, s, cl, 'M')
            vB = _cell_vector_3way(res, s, cl, 'F')
            est, se, z, p, d, lo, hi = contrast_stats(res, vA, vB)
            rows.append({
                'variable': var, 'contrast_type': 'within_cluster_within_scenario_Male_minus_Female',
                'cluster': cl, 'scenario': s, 'level_A': 'M', 'level_B': 'F',
                'estimate_A_minus_B': est, 'std_error': se, 'z_value': z,
                'p_value_raw': p, 'cohens_d': d, 'ci_low': lo, 'ci_high': hi
            })

contrasts_df = pd.DataFrame(rows)
mask = contrasts_df['p_value_raw'].notna()
_, p_adj, _, _ = multipletests(contrasts_df.loc[mask, 'p_value_raw'], method='fdr_bh')
contrasts_df.loc[mask, 'p_value_fdr'] = p_adj
contrasts_df['sig_plot'] = (contrasts_df['p_value_fdr'] < P_THRESH) & (contrasts_df['cohens_d'].abs() > D_THRESH)

#%% Omnibus Sex×Time (ML) — one line per variable
sex_time_rows = []

for var, res_full_reml in results.items():
    # keep the same random structure you used for var
    re_formula = _same_random_formula(res_full_reml)

    # FULL (ML): scenario * cluster * sex
    full_formula = f"{var} ~ scenario * C(cluster, Treatment(reference='Morning')) * C(sex, Treatment(reference='F'))"
    full_ml = _fit_ml(full_formula, data, re_formula)

    # REDUCED (ML): drop only the Sex×Time block (cluster:sex and scenario:cluster:sex)
    # keep main effects and the other 2-ways (scenario:cluster, scenario:sex)
    reduced_formula = (
        f"{var} ~ scenario + C(cluster, Treatment(reference='Morning')) + C(sex, Treatment(reference='F')) "
        f"+ scenario:C(cluster, Treatment(reference='Morning')) "
        f"+ scenario:C(sex, Treatment(reference='F'))"
    )
    red_ml = _fit_ml(reduced_formula, data, re_formula)

    # LR test
    lr_stat = 2.0 * (full_ml.llf - red_ml.llf)
    df_lrt  = max(0, len(full_ml.fe_params) - len(red_ml.fe_params))  # typically 4
    p_lr    = chi2.sf(lr_stat, df_lrt) if df_lrt > 0 else np.nan

    # R² (ML fits) and partial ΔR²m for the Sex×Time block
    r2m_full, r2c_full, var_re_full, resid_full = _r2_marginal_conditional(full_ml)
    r2m_red,  _,        _,            _         = _r2_marginal_conditional(red_ml)
    
    delta_r2m_raw = (r2m_full - r2m_red) if (pd.notna(r2m_full) and pd.notna(r2m_red)) else np.nan
    delta_r2m     = max(0.0, float(delta_r2m_raw)) if pd.notna(delta_r2m_raw) else np.nan
    
    f2 = (delta_r2m / (1 - r2m_full)) if (pd.notna(delta_r2m) and pd.notna(r2m_full) and r2m_full < 1) else np.nan
    f  = np.sqrt(f2) if (pd.notna(f2) and f2 >= 0) else np.nan


    # Cohen's f from ΔR²m
    f2 = (delta_r2m / (1 - r2m_full)) if (pd.notna(delta_r2m) and pd.notna(r2m_full) and r2m_full < 1) else np.nan
    f  = np.sqrt(f2) if (pd.notna(f2) and f2 >= 0) else np.nan

    # simple random-variance summary (sum of RE variances)
    try:
        rand_var = float(np.sum(np.diag(np.asarray(full_ml.cov_re))))
    except Exception:
        rand_var = np.nan

    sex_time_rows.append({
        "Variable": var,
        "LR stat (Sex×Time block)": lr_stat,
        "df": df_lrt,
        "p-value": p_lr,
        "Marginal R² (full, ML)": r2m_full,
        "Conditional R² (full, ML)": r2c_full,
        "ΔR²m due to Sex×Time (ML)": delta_r2m,
        "Cohen's f (Sex×Time, ML)": f,
        "Residual var. (full, ML)": resid_full,
        "Random var. (full, ML)": rand_var,
    })

sex_time_df = pd.DataFrame(sex_time_rows)

# Optional formatted copy for pretty Excel
sex_time_fmt = sex_time_df.copy()
for c in ["LR stat (Sex×Time block)", "Marginal R² (full, ML)", "Conditional R² (full, ML)",
          "ΔR²m due to Sex×Time (ML)", "Cohen's f (Sex×Time, ML)", "Residual var. (full, ML)", "Random var. (full, ML)"]:
    sex_time_fmt[c] = sex_time_fmt[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
sex_time_fmt["p-value"] = sex_time_df["p-value"].apply(lambda x: f"{x:.2E}" if pd.notna(x) else "")

excel_omnibus = os.path.join(OUT_DIR, "MixedLM_SexTime_Omnibus.xlsx")
with pd.ExcelWriter(excel_omnibus, engine="openpyxl") as w:
    sex_time_df.to_excel(w, sheet_name="numeric", index=False)
    sex_time_fmt.to_excel(w, sheet_name="formatted", index=False)
print(f"[OK] Wrote: {excel_omnibus}")

excel_contrasts = os.path.join(OUT_DIR, "MixedLM_SexTime_Contrasts.xlsx")
with pd.ExcelWriter(excel_contrasts, engine="openpyxl") as w:
    contrasts_df.to_excel(w, sheet_name="contrasts", index=False)
print(f"[OK] Wrote: {excel_contrasts}")

#%% Omnibus Sex×Time (ML) — one line per variable
sex_time_rows = []

for var, res_full_reml in results.items():
    # keep the same random structure you used for var
    re_formula = _same_random_formula(res_full_reml)

    # FULL (ML): scenario * cluster * sex
    full_formula = f"{var} ~ scenario * C(cluster, Treatment(reference='Morning')) * C(sex, Treatment(reference='F'))"
    full_ml = _fit_ml(full_formula, data, re_formula)

    # REDUCED (ML): drop only the Sex×Time block (cluster:sex and scenario:cluster:sex)
    # keep main effects and the other 2-ways (scenario:cluster, scenario:sex)
    reduced_formula = (
        f"{var} ~ scenario + C(cluster, Treatment(reference='Morning')) + C(sex, Treatment(reference='F')) "
        f"+ scenario:C(cluster, Treatment(reference='Morning')) "
        f"+ scenario:C(sex, Treatment(reference='F'))"
    )
    red_ml = _fit_ml(reduced_formula, data, re_formula)

    # LR test
    lr_stat = 2.0 * (full_ml.llf - red_ml.llf)
    df_lrt  = max(0, len(full_ml.fe_params) - len(red_ml.fe_params))  # typically 4
    p_lr    = chi2.sf(lr_stat, df_lrt) if df_lrt > 0 else np.nan

    # R² (ML fits) and partial ΔR²m for the interaction block
    r2m_full, r2c_full, var_re_full, resid_full = _r2_marginal_conditional(full_ml)
    r2m_red,  _,        _,            _         = _r2_marginal_conditional(red_ml)
    delta_r2m = (r2m_full - r2m_red) if (pd.notna(r2m_full) and pd.notna(r2m_red)) else np.nan

    # Cohen's f from ΔR²m
    f2 = (delta_r2m / (1 - r2m_full)) if (pd.notna(delta_r2m) and pd.notna(r2m_full) and r2m_full < 1) else np.nan
    f  = np.sqrt(f2) if (pd.notna(f2) and f2 >= 0) else np.nan

    # simple random-variance summary (sum of RE variances)
    try:
        rand_var = float(np.sum(np.diag(np.asarray(full_ml.cov_re))))
    except Exception:
        rand_var = np.nan

    sex_time_rows.append({
        "Variable": var,
        "LR stat (Sex×Time block)": lr_stat,
        "df": df_lrt,
        "p-value": p_lr,
        "Marginal R² (full, ML)": r2m_full,
        "Conditional R² (full, ML)": r2c_full,
        "ΔR²m due to Sex×Time (ML)": delta_r2m,
        "Cohen's f (Sex×Time, ML)": f,
        "Residual var. (full, ML)": resid_full,
        "Random var. (full, ML)": rand_var,
    })

sex_time_df = pd.DataFrame(sex_time_rows)

# Optional formatted copy for pretty Excel
sex_time_fmt = sex_time_df.copy()
for c in ["LR stat (Sex×Time block)", "Marginal R² (full, ML)", "Conditional R² (full, ML)",
          "ΔR²m due to Sex×Time (ML)", "Cohen's f (Sex×Time, ML)", "Residual var. (full, ML)", "Random var. (full, ML)"]:
    sex_time_fmt[c] = sex_time_fmt[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
sex_time_fmt["p-value"] = sex_time_df["p-value"].apply(lambda x: f"{x:.2E}" if pd.notna(x) else "")

excel_omnibus = os.path.join(OUT_DIR, "MixedLM_SexTime_Omnibus.xlsx")
with pd.ExcelWriter(excel_omnibus, engine="openpyxl") as w:
    sex_time_df.to_excel(w, sheet_name="numeric", index=False)
    sex_time_fmt.to_excel(w, sheet_name="formatted", index=False)
print(f"[OK] Wrote: {excel_omnibus}")


#%% Cell means for plotting 

cell_rows = []
for var, res in results.items():
    for sex in sexes:
        for cl in clusters:
            for sc in scenarios:
                est, se, lo, hi = cell_mean_ci(res, sc, cl, sex)
                cell_rows.append({
                    'variable': var, 'sex': sex, 'cluster': cl, 'scenario': sc,
                    'estimate': est, 'std_error': se, 'ci_low': lo, 'ci_high': hi
                })
cell_df = pd.DataFrame(cell_rows)

#%% PLOT A (rows=Sex; overlay Morning/Afternoon) -- not using this

fig, axes = plt.subplots(len(sexes), len(variable_settings), figsize=(18, 16))
fig.subplots_adjust(hspace=0.8)

for row_idx, sex_value in enumerate(sexes):
    for col_idx, variable in enumerate(variable_settings.keys()):
        ax = axes[row_idx, col_idx]
        s = variable_settings[variable]
        ylim, spacing, manual_offset, step = s['ylim'], s['spacing'], s['manual_offset'], s['gridline_step']

        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.grid(axis='y', linestyle="--", linewidth=0.5, color="#75726C", alpha=0.5)
        ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1);   ax.spines['bottom'].set_linewidth(1)


        # Plot errorbars for each cluster
        d = cell_df[(cell_df['variable'] == variable) & (cell_df['sex'] == sex_value)]
        x = np.arange(len(scenarios))
        for cl in clusters:
            dd = d[d['cluster'] == cl].set_index('scenario').loc[scenarios]
            x_positions = [xi + cluster_shift[cl] for xi in x]
            ax.errorbar(
                x_positions, dd['estimate'].values,
                yerr=[(dd['estimate'] - dd['ci_low']).values, (dd['ci_high'] - dd['estimate']).values],
                fmt=cluster_markers[cl], color=cluster_colors[cl],
                ecolor=cluster_colors[cl], capsize=1, markersize=1.5, elinewidth=1, label=cl
            )


        # Titles/labels (show titles only on first row)
        if row_idx == 0:
            ax.set_title(s['title'], fontsize=14)
        else:
            ax.set_title("")

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, fontsize=12)
        ax.set_title(s['title'], fontsize=14)
        if col_idx == 0:
            ax.set_ylabel(f"Sex: {sex_value}", fontsize=14)

        # Top bars: within-sex within-cluster across scenarios (no gaps)
        max_error_y = float(d['ci_high'].dropna().max())
        base_y      = max_error_y + manual_offset
        block_gap   = spacing  # keep one-spacing gap between color blocks
        
        sig_top = contrasts_df[
            (contrasts_df['variable'] == variable) &
            (contrasts_df['contrast_type'] == 'within_sex_within_cluster_across_scenarios') &
            (contrasts_df['sex'] == sex_value) &
            contrasts_df['sig_plot']
        ].copy()
        
        sig_top['x0'] = sig_top['level_A'].map({s: j for j, s in enumerate(scenarios)})
        sig_top['x1'] = sig_top['level_B'].map({s: j for j, s in enumerate(scenarios)})
        
        rows_m = sig_top[sig_top['cluster'] == 'Morning'].sort_values(['x0','x1'])
        rows_a = sig_top[sig_top['cluster'] == 'Afternoon'].sort_values(['x0','x1'])
        
        y = base_y + spacing
        y = _draw_sig_bars(ax, rows_m, cluster_colors['Morning'], cluster_shift['Morning'], y, spacing)
        y = _draw_sig_bars(ax, rows_a, cluster_colors['Afternoon'], cluster_shift['Afternoon'], y + block_gap, spacing)


        # Bottom labels: Afternoon − Morning within scenario (for this sex)
        sig_bot = contrasts_df[
            (contrasts_df['variable'] == variable) &
            (contrasts_df['contrast_type'] == 'within_sex_within_scenario_Afternoon_minus_Morning') &
            (contrasts_df['sex'] == sex_value) &
            contrasts_df['sig_plot']
        ]
        for _, r in sig_bot.iterrows():
            x_pos = scenarios.index(r['scenario'])
            ax.text(x_pos, ylim[0] - 0.05*(ylim[1]-ylim[0]), f"{r['cohens_d']:.2f}",
                    ha='center', va='top', fontsize=12, color='black')

        # Clean spines
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Legend (bottom)
fig.legend(
    handles=[
        Line2D([0], [0], color=cluster_colors["Morning"], marker=cluster_markers["Morning"], linestyle='', label='Morning'),
        Line2D([0], [0], color=cluster_colors["Afternoon"], marker=cluster_markers["Afternoon"], linestyle='', label='Afternoon'),
    ],
    loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2, fontsize=12, frameon=False
)

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "ErrorPlots_Cluster_Comparison.jpeg"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "ErrorPlots_Cluster_Comparison.pdf"), dpi=300)
plt.show()

#%% PLOT B (rows=Cluster; overlay Female/Male) 
fig, axes = plt.subplots(len(clusters), len(variable_settings), figsize=(18, 16))
fig.subplots_adjust(hspace=0.15, bottom=0.14)
# fig.subplots_adjust(hspace=0.2)

for row_idx, cluster_value in enumerate(clusters):
    for col_idx, variable in enumerate(variable_settings.keys()):
        ax = axes[row_idx, col_idx]
        s = variable_settings[variable]
        ylim, spacing, manual_offset, step = s['ylim'], s['spacing'], s['manual_offset'], s['gridline_step']

        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.grid(axis='y', linestyle="--", linewidth=0.5, color="#75726C", alpha=0.5)
        ax.spines['top'].set_visible(False);  ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1);   ax.spines['bottom'].set_linewidth(1)

        # Plot errorbars for each sex
        d = cell_df[(cell_df['variable'] == variable) & (cell_df['cluster'] == cluster_value)]
        x = np.arange(len(scenarios))
        for sex in sexes:
            dd = d[d['sex'] == sex].set_index('scenario').loc[scenarios]
            # Plot B — inside: for sex in sexes:
            x_positions = [xi + sex_shift[sex] for xi in x]
            ax.errorbar(
                x_positions, dd['estimate'].values,
                yerr=[(dd['estimate'] - dd['ci_low']).values, (dd['ci_high'] - dd['estimate']).values],
                fmt=sex_markers[sex],                 # <-- was sex_markers[cl]
                color=sex_colors[sex],                # <-- was sex_colors[cl]
                ecolor=sex_colors[sex],               # <-- was sex_colors[cl]
                capsize=4, markersize=4, elinewidth=1,
                label=sex
            )



        # Titles/labels (show titles only on first row)
        if row_idx == 0:
            ax.set_title(s['title'], fontsize=14)
        else:
            ax.set_title("")

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, fontsize=12)
        # ax.set_title(s['title'], fontsize=14)
        if col_idx == 0:
            ax.set_ylabel(f"Cluster: {cluster_value}", fontsize=14)

        # Top bars: within-cluster across scenarios by sex (no gaps)
        max_error_y = float(d['ci_high'].dropna().max())
        base_y      = max_error_y + manual_offset
        block_gap   = spacing
        
        sig_top = contrasts_df[
            (contrasts_df['variable'] == variable) &
            (contrasts_df['contrast_type'] == 'within_sex_within_cluster_across_scenarios') &
            (contrasts_df['cluster'] == cluster_value) &
            contrasts_df['sig_plot']
        ].copy()
        
        sig_top['x0'] = sig_top['level_A'].map({s: j for j, s in enumerate(scenarios)})
        sig_top['x1'] = sig_top['level_B'].map({s: j for j, s in enumerate(scenarios)})
        
        rows_f = sig_top[sig_top['sex'] == 'F'].sort_values(['x0','x1'])
        rows_m = sig_top[sig_top['sex'] == 'M'].sort_values(['x0','x1'])
        
        y = base_y + spacing
        y = _draw_sig_bars(ax, rows_f, sex_colors['F'], sex_shift['F'], y, spacing)
        y = _draw_sig_bars(ax, rows_m, sex_colors['M'], sex_shift['M'], y + block_gap, spacing)


        # Bottom labels: Male − Female within scenario (for this cluster)
        sig_bot = contrasts_df[
            (contrasts_df['variable'] == variable) &
            (contrasts_df['contrast_type'] == 'within_cluster_within_scenario_Male_minus_Female') &
            (contrasts_df['cluster'] == cluster_value) &
            contrasts_df['sig_plot']
        ]
        for _, r in sig_bot.iterrows():
            x_pos = scenarios.index(r['scenario'])
            ax.text(x_pos, ylim[0] - 0.05*(ylim[1]-ylim[0]), f"{r['cohens_d']:.2f}",
                    ha='center', va='top', fontsize=12, color='black')

        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Legend
fig.legend(
    handles=[
        Line2D([0], [0], color=sex_colors["F"], marker=sex_markers["F"], linestyle='', label='Female'),
        Line2D([0], [0], color=sex_colors["M"], marker=sex_markers["M"], linestyle='', label='Male'),
    ],
    loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2, fontsize=14, frameon=False
)

# plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, "ErrorPlots_Sex_Comparison.jpeg"), dpi=300)
plt.savefig(os.path.join(OUT_DIR, "ErrorPlots_Sex_Comparison.pdf"), dpi=300)
plt.show()
