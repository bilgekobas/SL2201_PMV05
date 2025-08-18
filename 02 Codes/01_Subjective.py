# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 23:24:44 2025

@author: kobas
"""

# -*- coding: utf-8 -*-
"""
Boxplots of TSV, TCV, KSS, and Thermal Satisfaction (VAS 1–101)
by Sex (rows) × Cluster (cols) × Scenario (x).

- Female boxes: #121417 (black-ish), Male boxes: #FF1F62 (pink) because why not.
- Median lines use the opposite color.
- Full y-scales; ref-lines: TSV=0, KSS=6, TSAT=50 (none for TCV).
- Horizontal dashed y-grid. No legend.
- Scenario order: T1, T2, ... (numeric if present).
- Paired Wilcoxon for within-panel scenario pairs (effect size d_z), annotated on plots (|effect|≥0.45 and p<=alpha).
- Mann–Whitney d for Female vs Male at (scenario, cluster) [CSV only].
- Paired Wilcoxon for Morning vs Afternoon within (sex, scenario) [CSV only].
- Exports:
    * 05_Subjective/<VAR>_Boxplots_SexClusterScenario_Thin.{jpeg,pdf}
    * 05_Subjective/Boxplot_Pairwise_Stats_ALL.csv
    * 05_Subjective/Boxplot_Medians_ByPanel.csv
"""

import os, warnings, re
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from scipy import stats

from pathlib import Path

def get_repo_root() -> Path:
    # Use __file__ when running as a script; fall back to CWD in notebooks
    here = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    # If this file is inside 02_Codes/, the repo root is one level up
    return here.parent.parent if here.parent.name == "02_Codes" else here

ROOT  = get_repo_root()
DATA  = ROOT / "01_Data"
GRAPHS = ROOT / "03_Graphs"
OUT   = ROOT / "04_Model_Outputs"
FONTS = ROOT / "Fonts"

# ---------------- YOU: set CWD + optional font ----------------
warnings.filterwarnings("ignore", message=".*findfont:.*")

FONTS = Path(__file__).resolve().parents[1] / "Fonts"
font_path = FONTS / "Archivo_SemiCondensed-Medium.ttf"

try:
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        custom_font = FontProperties(fname=str(font_path))
        rcParams["font.family"] = custom_font.get_name()
    else:
        custom_font = None
except Exception:
    custom_font = None

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype']  = 42

# ---------------- paths ----------------
# DATA_FILE_CSV = DATA / '00_SL2201_SubjectiveResponses.csv'
DATA_FILE_XLSX = DATA / '00_SL2201_SubjectiveResponses.xlsx'
OUT_DIR = ROOT / '05_Subjective'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- config (styling + labels) ----------------
COLOR_F = "#121417"     # Female
COLOR_M = "#FF1F62"     # Male

FONTS = {               # quick global sizes (you can tweak)
    "suptitle": 12,
    "panel_title": 10,
    "xtick": 8,
    "ytick": 7,
    "ann": 7,
}

# reference lines per variable
REFLINES = {
    "tsv": 0,
    "tcv": None,
    "kss": 6,
    "tsat": 50,   # Thermal Satisfaction (VAS 1–101)
}

# textual y-ticks where appropriate
SELECTIVE_LABELS = {
    "tsv": {-3: "Cold", 0: "Neutral", 3: "Hot"},
    "tcv": {0: "Comfortable", 4: "Extremely\nuncomfortable"},
    "kss": {1: "Extremely\nsleepy", 6: "Neither alert\nnor sleepy", 10: "Extremely\nalert"},
    # TSAT (VAS): numeric ticks only
}

# full y ranges
YLIMS = {
    "tsv":  (-3,  3),
    "tcv":  ( 0,  4),
    "kss":  ( 1, 10),
    "tsat": ( 1,101),   # VAS scale 1..101
}

# tick steps for grids
YMAJOR_STEP = {
    "tsv": 1,
    "tcv": 1,
    "kss": 1,
    "tsat": 10,
}

# subject ID column candidates
SUBJECT_ID_CANDIDATES = ["part-id", "participant-id", "participant", "subject", "id"]

# --------------- helpers: load, normalize, tidy ----------------
def norm(c: str) -> str:
    return str(c).strip().lower().replace(" ", "-").replace("_", "-")

def load_table_flexible(csv_path: Path, xlsx_path: Path):
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if xlsx_path.exists():
        return pd.read_excel(xlsx_path, sheet_name=0, engine='openpyxl')
    raise FileNotFoundError('Neither CSV nor XLSX subjective data file found in 01_Data/')

df = load_table_flexible(DATA_FILE_XLSX)
df.columns = [norm(c) for c in df.columns]

# map sex / cluster
def map_sex(x):
    x = str(x).strip().lower()
    if x.startswith("f"): return "Female"
    if x.startswith("m"): return "Male"
    return x

def map_cluster(x):
    x = str(x).strip().lower()
    if "morn" in x: return "Morning"
    if "after" in x: return "Afternoon"
    return x

# detect subject id col
SUBJECT_ID_COL = None
for cand in SUBJECT_ID_CANDIDATES:
    if cand in df.columns:
        SUBJECT_ID_COL = cand
        break

print("Subject ID col:", SUBJECT_ID_COL if SUBJECT_ID_COL else "NOT FOUND (paired tests may be disabled)")

# required columns (we’ll find TSAT flexibly)
required = ["scenario", "sex", "cluster"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# numeric columns (core)
num_cols = []
for base in ["tsv", "tcv", "kss"]:
    if base in df.columns:
        num_cols.append(base)

# thermal satisfaction detection (tsat)
tsat_col = None
for c in df.columns:
    if re.search(r"\b(tsat|thermal-?satisf)", c):
        tsat_col = c
        break
if tsat_col:
    if tsat_col != "tsat":
        df = df.rename(columns={tsat_col: "tsat"})
    num_cols.append("tsat")

if not num_cols:
    raise ValueError("No numeric questionnaire columns found (tsv/tcv/kss/tsat).")

# tidy
df = df.dropna(subset=["scenario", "sex", "cluster"]).copy()
df["sex"] = df["sex"].map(map_sex)
df["cluster"] = df["cluster"].map(map_cluster)
for v in num_cols:
    df[v] = pd.to_numeric(df[v], errors="coerce")

# scenario order (T1..Tn numeric)
def scen_key(s):
    m = re.search(r'(\d+)', str(s))
    return (0, int(m.group(1))) if m else (1, str(s))
SCENARIOS = sorted(pd.unique(df["scenario"]).tolist(), key=scen_key)

# --------------- helpers: stats & ticks ----------------
def build_yticks(var: str):
    lo, hi = YLIMS[var]
    step = YMAJOR_STEP[var]
    ticks = list(range(int(lo), int(hi)+1, step))
    labels = []
    sel = SELECTIVE_LABELS.get(var, {})
    for k in ticks:
        if k in sel:
            labels.append(f"{int(k)}\n{sel[k]}")  # two-line label
        else:
            labels.append(str(int(k)))
    return ticks, labels

def pooled_sd(a: np.ndarray, b: np.ndarray):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp2 = ((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2)
    return np.sqrt(sp2) if sp2 > 0 else np.nan

def cohens_d_ind(a, b):
    a = np.asarray(pd.to_numeric(a, errors="coerce"), float); a = a[~np.isnan(a)]
    b = np.asarray(pd.to_numeric(b, errors="coerce"), float); b = b[~np.isnan(b)]
    sp = pooled_sd(a, b)
    if sp == 0 or np.isnan(sp): return np.nan
    return (np.mean(a) - np.mean(b)) / sp

def dz_paired(diff):
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 2: return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0: return np.nan
    return np.mean(diff) / sd

def holm_adjust(pvals):
    p = np.array([1.0 if (pv is None or (isinstance(pv, float) and np.isnan(pv))) else pv for pv in pvals], dtype=float)
    order = np.argsort(p); m = len(p)
    adj = np.ones_like(p, dtype=float)
    for r, idx in enumerate(order, start=1):
        adj[idx] = min((m - r + 1) * p[idx], 1.0)
    return adj

def draw_sig_bar(ax, x1, x2, y, text, frac=0.018, color="0.25", fontsize=7):
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = frac * yr
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, color=color)
    ax.text((x1+x2)/2, y+h, text, ha="center", va="bottom", fontsize=fontsize, color=color)

# master collectors
ALL_ROWS = []
MEDIAN_ROWS = []

# --------------- plotting (with paired scenario annotations) ---------------
def plot_one(
    var: str,
    title: str,
    fname_base: str,
    box_width=0.06,
    spacing=0.70,          # tighter spacing → smaller gap between scenarios
    edge_pad_frac=0.10,    # shrink left/right padding to trim empty edges
    figsize=(5.0, 3.6),
    dpi=300,
    annotation_p="holm",   # "holm" or "raw" for labels
    alpha=0.05,
    effect_thr=0.45,       # annotate only if |effect| >= this
):
    yticks, yticklbls = build_yticks(var)
    ylim = YLIMS[var]
    refline = REFLINES.get(var, None)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=FONTS["suptitle"])

    # rows: Female, Male ; cols: Morning, Afternoon
    row_order = ["Female", "Male"]
    col_order = ["Morning", "Afternoon"]

    for i, sex in enumerate(row_order):
        for j, cluster in enumerate(col_order):
            ax = axes[i, j]
            sub = df[(df["sex"]==sex) & (df["cluster"]==cluster)]

            # positions/data
            x = np.arange(len(SCENARIOS), dtype=float) * spacing
            panels = []
            for sc in SCENARIOS:
                vals = sub[sub["scenario"]==sc][var].dropna().values
                panels.append(vals)

            # grid
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(MultipleLocator(YMAJOR_STEP[var]))
            ax.grid(axis='y', which='major', color='#2d2d2d', linewidth=0.45,
                    linestyle="--", alpha=0.85, zorder=0)

            # boxplots
            bp = ax.boxplot(panels, positions=x, widths=box_width, patch_artist=True, showfliers=False)
            box_color = COLOR_F if sex=="Female" else COLOR_M
            med_color = COLOR_M if sex=="Female" else COLOR_F
            for patch in bp['boxes']:
                patch.set_facecolor(box_color)
                patch.set_edgecolor(box_color)
                patch.set_alpha(0.85 if sex=="Male" else 0.75)
                patch.set_linewidth(1.0)
            for med in bp['medians']:
                med.set_linewidth(1.7)
                med.set_color(med_color)
            for w in bp['whiskers'] + bp['caps']:
                w.set_color(box_color); w.set_linewidth(1.0)

            # medians table collector
            for idx, sc in enumerate(SCENARIOS):
                arr = panels[idx]
                med = float(np.nanmedian(arr)) if len(arr) else np.nan
                MEDIAN_ROWS.append({
                    "variable": var, "sex": sex, "cluster": cluster, "scenario": sc,
                    "n": int(len(arr)), "median": med
                })

            # x-limits (trim side space)
            if len(x):
                pad = edge_pad_frac * spacing
                ax.set_xlim(x[0] - pad, x[-1] + pad)

            # axes cosmetics
            ax.set_title(f"{sex} — {cluster}", fontsize=FONTS["panel_title"])
            ax.set_xticks(x)
            have = [len(p) > 0 for p in panels]
            ax.set_xticklabels([str(SCENARIOS[k]) if have[k] else "" for k in range(len(SCENARIOS))],
                               fontsize=FONTS["xtick"])
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklbls, fontsize=FONTS["ytick"], linespacing=0.9)
            ax.tick_params(axis='x', labelsize=FONTS["xtick"])
            ax.tick_params(axis='y', labelsize=FONTS["ytick"])
            for spine in ("top","right"): ax.spines[spine].set_visible(False)

            # reference horizontal line
            if refline is not None and (ylim[0] <= refline <= ylim[1]):
                ax.axhline(refline, lw=0.8, color="black", zorder=2)

            # ---- scenario-within (paired Wilcoxon) + annotate ----
            pairs = []
            stats_list = []
            pvals = []
            effs = []
            ns_aligned = []
            means_a = []; means_b = []; med_a = []; med_b = []

            for a_idx, b_idx in combinations(range(len(SCENARIOS)), 2):
                scA, scB = SCENARIOS[a_idx], SCENARIOS[b_idx]
                A = sub[sub["scenario"]==scA]
                B = sub[sub["scenario"]==scB]
                if SUBJECT_ID_COL and SUBJECT_ID_COL in A.columns and SUBJECT_ID_COL in B.columns:
                    merged = pd.merge(
                        A[[SUBJECT_ID_COL, var]],
                        B[[SUBJECT_ID_COL, var]],
                        on=SUBJECT_ID_COL, how="inner", suffixes=("_A", "_B")
                    ).dropna()
                    if not merged.empty:
                        diffs = merged[f"{var}_A"].values - merged[f"{var}_B"].values
                        # Wilcoxon; skip if all diffs are zero
                        if np.allclose(diffs, 0):
                            stat, p = np.nan, np.nan
                            d_eff = np.nan
                            n_pairs = len(diffs)
                        else:
                            stat, p = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")
                            d_eff = dz_paired(diffs)
                            n_pairs = int(np.count_nonzero(np.abs(diffs) > 0))
                        n_a = int(A[var].notna().sum()); n_b = int(B[var].notna().sum())
                        mu_a = float(np.nanmean(A[var])); mu_b = float(np.nanmean(B[var]))
                        me_a = float(np.nanmedian(A[var])); me_b = float(np.nanmedian(B[var]))
                    else:
                        stat=p=np.nan; d_eff=np.nan
                        n_a=n_b=n_pairs=0; mu_a=mu_b=me_a=me_b=np.nan
                else:
                    # no subject ID — cannot do paired test; fall back to NaNs
                    stat=p=np.nan; d_eff=np.nan
                    n_a=n_b=n_pairs=0; mu_a=mu_b=me_a=me_b=np.nan

                pairs.append((a_idx, b_idx, scA, scB))
                stats_list.append(stat); pvals.append(p); effs.append(d_eff); ns_aligned.append(n_pairs)
                means_a.append(mu_a); means_b.append(mu_b); med_a.append(me_a); med_b.append(me_b)

            if pairs:
                p_holm = holm_adjust([pv if not np.isnan(pv) else 1.0 for pv in pvals])
                used_p = p_holm if annotation_p=="holm" else pvals
                p_label = "p(Holm)" if annotation_p=="holm" else "p"

                # annotate selected pairs
                y_base = ylim[1] - 0.06*(ylim[1]-ylim[0])
                step   = 0.05*(ylim[1]-ylim[0])
                drawn  = 0
                for (a_idx,b_idx,scA,scB), p_raw, p_adj, p_used, d_eff in zip(
                    pairs, pvals, p_holm, used_p, effs
                ):
                    if np.isfinite(p_used) and np.isfinite(d_eff) and (p_used <= alpha) and (abs(d_eff) >= effect_thr):
                        y = y_base - drawn*step
                        draw_sig_bar(ax, x[a_idx], x[b_idx], y,
                                     f"dz={d_eff:.2f}, {p_label}={p_used:.3g}",
                                     frac=0.015, fontsize=FONTS["ann"])
                        drawn += 1

                # export rows for this family
                for (a_idx,b_idx,scA,scB), stat, p, adj, d_eff, n_pairs, ma, mb, me_a, me_b in zip(
                    pairs, stats_list, pvals, p_holm, effs, ns_aligned, means_a, means_b, med_a, med_b
                ):
                    ALL_ROWS.append({
                        "comparison": "scenario_within_sex_cluster",
                        "test": "wilcoxon",
                        "effect": "dz",
                        "variable": var, "sex": sex, "cluster": cluster,
                        "scenario_a": scA, "scenario_b": scB,
                        "n_a": sub[sub["scenario"]==scA][var].notna().sum(),
                        "n_b": sub[sub["scenario"]==scB][var].notna().sum(),
                        "n_pairs": n_pairs,
                        "mean_a": ma, "mean_b": mb,
                        "median_a": me_a, "median_b": me_b,
                        "stat": stat, "p_value": p, "p_holm": adj, "effect_size": d_eff
                    })

    fig.subplots_adjust(wspace=0.25, hspace=0.35)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    jpg = OUT_DIR / f"{fname_base}.jpeg"
    pdf = OUT_DIR / f"{fname_base}.pdf"
    fig.savefig(jpg, dpi=dpi, facecolor="white")
    fig.savefig(pdf, dpi=dpi, facecolor="white")
    plt.show()

# --------------- extra comparisons for CSV (no annotations) ---------------
def add_sex_diffs_same_scen_cluster(var: str):
    # Female vs Male at each (scenario, cluster) — independent Mann–Whitney, Cohen's d
    for cluster in ["Morning", "Afternoon"]:
        for scen in SCENARIOS:
            A = df[(df["cluster"]==cluster) & (df["scenario"]==scen) & (df["sex"]=="Female")][var].dropna()
            B = df[(df["cluster"]==cluster) & (df["scenario"]==scen) & (df["sex"]=="Male")][var].dropna()
            if len(A)==0 or len(B)==0:
                stat = p = d = np.nan
            else:
                stat, p = stats.mannwhitneyu(A, B, alternative="two-sided")
                d = cohens_d_ind(A, B)
            ALL_ROWS.append({
                "comparison": "sex_within_scenario_cluster",
                "test": "mannwhitney",
                "effect": "d",
                "variable": var, "sex": "Female_vs_Male", "cluster": cluster,
                "scenario_a": scen, "scenario_b": scen,
                "n_a": len(A), "n_b": len(B), "n_pairs": "",
                "mean_a": float(np.nanmean(A)) if len(A) else np.nan,
                "mean_b": float(np.nanmean(B)) if len(B) else np.nan,
                "median_a": float(np.nanmedian(A)) if len(A) else np.nan,
                "median_b": float(np.nanmedian(B)) if len(B) else np.nan,
                "stat": stat, "p_value": p, "p_holm": p if np.isfinite(p) else np.nan, "effect_size": d
            })

def add_cluster_diffs_same_sex_scen(var: str):
    # Morning vs Afternoon within (sex, scenario) — paired Wilcoxon, dz
    if not SUBJECT_ID_COL or SUBJECT_ID_COL not in df.columns:
        return
    for sex in ["Female", "Male"]:
        for scen in SCENARIOS:
            A = df[(df["sex"]==sex) & (df["scenario"]==scen) & (df["cluster"]=="Morning")][[SUBJECT_ID_COL, var]]
            B = df[(df["sex"]==sex) & (df["scenario"]==scen) & (df["cluster"]=="Afternoon")][[SUBJECT_ID_COL, var]]
            merged = pd.merge(A, B, on=SUBJECT_ID_COL, how="inner", suffixes=("_M", "_A")).dropna()
            if not merged.empty:
                diffs = merged[f"{var}_M"].values - merged[f"{var}_A"].values
                if np.allclose(diffs, 0):
                    stat = p = np.nan
                    d = np.nan
                    n_pairs = len(diffs)
                else:
                    stat, p = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")
                    d = dz_paired(diffs); n_pairs = int(np.count_nonzero(np.abs(diffs) > 0))
            else:
                stat=p=d=np.nan; n_pairs=0
            ALL_ROWS.append({
                "comparison": "cluster_within_sex_scenario",
                "test": "wilcoxon",
                "effect": "dz",
                "variable": var, "sex": sex, "cluster": "Morning_vs_Afternoon",
                "scenario_a": scen, "scenario_b": scen,
                "n_a": df[(df["sex"]==sex) & (df["scenario"]==scen) & (df["cluster"]=="Morning")][var].notna().sum(),
                "n_b": df[(df["sex"]==sex) & (df["scenario"]==scen) & (df["cluster"]=="Afternoon")][var].notna().sum(),
                "n_pairs": n_pairs,
                "mean_a": float(np.nanmean(A[var])) if len(A) else np.nan,
                "mean_b": float(np.nanmean(B[var])) if len(B) else np.nan,
                "median_a": float(np.nanmedian(A[var])) if len(A) else np.nan,
                "median_b": float(np.nanmedian(B[var])) if len(B) else np.nan,
                "stat": stat, "p_value": p, "p_holm": p if np.isfinite(p) else np.nan, "effect_size": d
            })

# --------------- RUN: four figures ---------------
# TSV
if "tsv" in df.columns:
    plot_one(
        var="tsv",
        title="Thermal Sensation Vote (TSV)",
        fname_base="TSV_Boxplots_SexClusterScenario_Thin",
        box_width=0.06, spacing=0.72, edge_pad_frac=0.10,
        figsize=(5.0, 3.6),
        annotation_p="holm", alpha=0.05, effect_thr=0.45
    )

# TCV
if "tcv" in df.columns:
    plot_one(
        var="tcv",
        title="Thermal Comfort Vote (TCV)",
        fname_base="TCV_Boxplots_SexClusterScenario_Thin",
        box_width=0.06, spacing=0.72, edge_pad_frac=0.10,
        figsize=(5.0, 3.6),
        annotation_p="holm", alpha=0.05, effect_thr=0.45
    )

# KSS
if "kss" in df.columns:
    plot_one(
        var="kss",
        title="Karolinska Sleepiness Scale (KSS)",
        fname_base="KSS_Boxplots_SexClusterScenario_Thin",
        box_width=0.06, spacing=0.72, edge_pad_frac=0.10,
        figsize=(5.0, 3.6),
        annotation_p="holm", alpha=0.05, effect_thr=0.45
    )

# Thermal Satisfaction (VAS 1–101)
if "tsat" in df.columns:
    plot_one(
        var="tsat",
        title="Thermal Satisfaction (VAS 1–101)",
        fname_base="TSAT_Boxplots_SexClusterScenario_Thin",
        box_width=0.06, spacing=0.72, edge_pad_frac=0.10,
        figsize=(5.0, 3.6),
        annotation_p="holm", alpha=0.05, effect_thr=0.45
    )

# --------------- Add non-annotated families to CSV ---------------
for v in [c for c in ["tsv", "tcv", "kss", "tsat"] if c in df.columns]:
    add_sex_diffs_same_scen_cluster(v)
    add_cluster_diffs_same_sex_scen(v)

# Save CSVs
pairs_df = pd.DataFrame(ALL_ROWS)
pairs_out = OUT_DIR / "Boxplot_Pairwise_Stats_ALL.csv"
# pairs_df.to_csv(pairs_out, index=False)

med_df = pd.DataFrame(MEDIAN_ROWS)
med_out = OUT_DIR / "Boxplot_Medians_ByPanel.csv"
med_df.to_csv(med_out, index=False)

print("Saved pairwise stats to:", pairs_out.resolve())
print("Saved medians to:", med_out.resolve())
print("Done.")


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

