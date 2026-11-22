"""Participant-blocked scenario contrasts for the PMV05 repeated-measures study."""
from html import escape
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Submission" / "Data in Brief"
OUT = ROOT / "Graphs" / "Reworked Results" / "Repeated measures analysis.html"
KEY = ["part-id", "scenario", "cluster"]
CONDITIONS = ["T1", "T2", "T3", "T4"]
PERIODS = ["Morning", "Afternoon"]
FAMILIES = {
    "PMV and SET": {"pmv": "PMV", "SET_C": "SET (°C)"},
    "Measured thermal exposure": {
        "tdb": "Air temperature (°C)", "tg": "Globe temperature (°C)",
        "mrt": "Mean radiant temperature (°C)", "top": "Operative temperature (°C)",
        "rh": "Relative humidity (percentage points)",
    },
    "Other exposure indicators": {
        "ppd": "PPD (percentage points)", "co2": "CO₂ (ppm)",
        "iaq": "Recorded IAQ index", "texp": "Recorded texp (°C)",
    },
    "Contextual temperatures": {
        "tdb-climalab": "Climalab air temperature (°C)", "tout-measured": "Outdoor temperature (°C)",
    },
    "Physiology": {
        "skin-temp-wrist-smooth": "Wrist skin temperature (°C)", "eda_tonic": "Tonic EDA",
        "hr-clean": "Heart rate (bpm)", "rmssd": "RMSSD", "lf/hf": "LF/HF",
    },
    "Questionnaire": {"tsv": "Thermal sensation vote", "tcv": "Thermal comfort vote"},
}
LOG = {"eda_tonic", "rmssd", "lf/hf"}
ORDINAL = {"tsv", "tcv"}
RNG = np.random.default_rng(2201)


def paired_estimate(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    assert 1 < n <= 11 and np.isfinite(values).all()
    if np.allclose(values, 0, rtol=0, atol=1e-12):
        return 0., 0., 0., 1.
    # Exact enumeration of participant-level sign flips under a symmetric null.
    signs = 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) - 1
    effect = values.mean()
    p = np.mean(np.abs(signs @ values / n) >= abs(effect) - 1e-12)
    bootstrap = values[RNG.integers(n, size=(20000, n))].mean(axis=1)
    lo, hi = np.quantile(bootstrap, [.025, .975])
    return effect, lo, hi, p


def holm(p):
    ordered = p.sort_values()
    adjusted = (ordered * np.arange(len(p), 0, -1)).cummax().clip(upper=1)
    return adjusted.reindex(p.index)


def visit_adjusted(values, order_delta):
    if order_delta.nunique() < 2:
        return np.nan
    design = np.column_stack([np.ones(len(values)), order_delta])
    return np.linalg.lstsq(design, values, rcond=None)[0][0]


def describe(session, visits):
    results = []
    counts = []
    for family, outcomes in FAMILIES.items():
        for key, label in outcomes.items():
            values = session[key].unstack(["scenario", "cluster"])
            counts.append({"Outcome": label, **{
                f"{s} {p}": f"{values[s, p].count()} / {values[s, p].median():.3g}"
                for s in CONDITIONS for p in PERIODS
            }})
            if key in LOG:
                assert (values.stack(future_stack=True).stack(future_stack=True).dropna() > 0).all().all()
                values = np.log(values)
            for a, b in combinations(CONDITIONS, 2):
                differences = pd.DataFrame({period: values[b, period] - values[a, period] for period in PERIODS})
                if key in ORDINAL:
                    differences = np.sign(differences)
                paired = differences.mean(axis=1).dropna()
                order = visits[b].loc[paired.index] - visits[a].loc[paired.index]
                estimate = visit_adjusted(paired, order)
                results.append(record(family, key, label, "Scenario", f"{b} − {a}", paired,
                                      differences.notna().all(axis=1).sum(), estimate))
                if key not in ORDINAL:
                    change = (differences["Afternoon"] - differences["Morning"]).dropna()
                    results.append(record(family, key, label, "Scenario by time", f"{b} − {a}", change, len(change)))
            for sc in CONDITIONS:
                am_pm = (values[sc, "Afternoon"] - values[sc, "Morning"]).dropna()
                if key in ORDINAL:
                    am_pm = np.sign(am_pm)
                results.append(record(family, key, label, "Time within scenario", sc, am_pm, len(am_pm)))
    results = pd.DataFrame(results)
    results["Holm p"] = results.groupby(["Family", "Comparison"])["p"].transform(holm)
    return results, pd.DataFrame(counts)


def record(family, key, label, kind, contrast, values, complete, adjusted=np.nan):
    estimate, lo, hi, p = paired_estimate(values)
    jackknife = (values.sum() - values) / (len(values) - 1)
    transform = np.exp if key in LOG else float
    return {"Family": family, "Outcome": label, "Comparison": kind, "Contrast": contrast,
            "Scale": "Ratio" if key in LOG else "Net higher-vote probability" if key in ORDINAL else "Difference",
            "n": len(values), "n both periods": complete if kind == "Scenario" else np.nan,
            "Estimate": transform(estimate), "95% CI low": transform(lo), "95% CI high": transform(hi),
            "p": p, "Order-adjusted estimate": transform(adjusted),
            "Leave-one-out low": transform(jackknife.min()), "Leave-one-out high": transform(jackknife.max())}


def table(data):
    return data.to_html(index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.4g}")


def report(results, counts, order, fixed):
    sections = [
        """<h1>Repeated measures analysis</h1>
<p>Four office scenarios, 11 participants, 44 visits and 88 in-lab sessions. Morning and afternoon were recorded on the same visit. Breaks were excluded. Climate and physiology inputs are saved session medians. Thermal sensation and comfort use the 83 observed questionnaire records. Higher TSV means warmer and higher TCV means more discomfort.</p>
<h2>Models and interpretation</h2>
<p>For continuous outcomes the participant-blocked contrast model estimates the mean of within-participant differences. Scenario contrasts average the morning and afternoon differences with equal weight. Afternoon–morning differences are estimated within scenario. A scenario-by-time contrast compares the afternoon–morning change between two scenarios. Tonic EDA, RMSSD and LF/HF are modeled on the log scale and reported as ratios of geometric means of matched session values. A ratio of 1 is the null.</p>
<p>Questionnaire comparisons model paired direction: each higher, tied or lower vote is coded +1, 0 or −1. The effect is the net probability of a higher vote, giving participants equal weight and using available same-time pairs for scenario comparisons. This preserves the order of categories without assuming equal distances between response options. Afternoon–morning comparisons use complete within-scenario pairs. A zero effect is the null for signed vote probabilities and continuous differences.</p>
<p>Two-sided p values enumerate all participant-level sign flips under a symmetric null distribution. These are model-based tests, with no assumption that visit order was randomized. The 95% intervals are unadjusted percentile intervals from 20,000 participant bootstrap samples and are approximate with 11 participants. Holm adjustment is applied across every outcome and contrast within each outcome family and comparison type. PMV and SET form the family directly addressing the common-target question. PPD, PMV and SET share inputs and cannot provide independent confirmations. Leave-one-out columns show the range of estimates after excluding one participant at a time.</p>
<p>For scenario contrasts, the order sensitivity estimate comes from regressing the participant contrast on the corresponding difference in visit numbers and evaluating at zero. It assumes a linear visit effect, is descriptive, and can extrapolate where one scenario consistently precedes another. The imbalance below limits causal separation of scenario, order and day. The afternoon–morning contrast measures the protocol time difference and may include cumulative exposure or other within-visit effects. This analysis estimates session-level responses and does not test within-session trajectories or sex moderation. No equivalence margins were specified.</p>
<h2>Visit order</h2>""",
        table(order.reset_index()),
        "<h2>Fixed exposure inputs</h2><p>Recorded air speed, clothing insulation and metabolic rate were constant within scenario in the analysis data. Values are reported directly. The file cannot establish within-scenario variability for these inputs.</p>",
        table(fixed.reset_index()),
        "<h2>Observed session counts and medians</h2><p>Cells are n / median in the original units.</p>",
        table(counts),
    ]
    for (family, kind), block in results.groupby(["Family", "Comparison"], sort=False):
        if kind != "Scenario":
            block = block.drop(columns=["n both periods", "Order-adjusted estimate"])
        sections += [f"<h2>{escape(family)} · {escape(kind)}</h2>",
                     table(block.drop(columns=["Family", "Comparison"]))]
    style = """body{font:15px/1.45 Arial,sans-serif;color:#182d38;max-width:1360px;margin:36px auto;padding:0 24px}
    p{max-width:96ch}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{padding:6px 8px;text-align:right;border-bottom:1px solid #dce5ea}th{background:#e8f0f4}
    th:first-child,td:first-child{text-align:left}tbody tr:nth-child(even){background:#f5f8fa}"""
    OUT.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Repeated measures analysis</title>'
                   f"<style>{style}</style><main>{''.join(sections)}</main></html>", encoding="utf-8")


def main():
    raw = pd.read_csv(DATA / "SET_timeseries_with_SET.csv")
    raw = raw.loc[raw["lab-id"].eq("SL") & raw["location"].eq("Lab") & raw["break"].ne(1)]
    visits = raw[["part-id", "scenario", "date"]].drop_duplicates()
    assert len(raw) == 15797 and len(visits) == 44
    visits["visit"] = pd.to_datetime(visits["date"], format="%d-%m-%y").groupby(visits["part-id"]).rank()
    order = pd.crosstab(visits["scenario"], visits["visit"].astype(int))
    visits = visits.pivot(index="part-id", columns="scenario", values="visit")
    fixed = raw.groupby("scenario")[["v", "clo", "met"]]
    assert fixed.nunique().eq(1).all().all()
    session = raw.groupby(KEY).median(numeric_only=True)
    votes = pd.read_excel(DATA / "00_SL2201_SubjectiveResponses.xlsx", header=1)
    votes = votes[KEY + ["tsv", "tcv"]].dropna(subset=KEY).copy()
    votes["part-id"] = "SL2201_" + votes["part-id"].str.strip()
    votes["cluster"] = votes["cluster"].str.split("_").str[-1]
    votes = votes.set_index(KEY)
    assert len(session) == 88 and len(votes) == 83 and votes.index.is_unique and votes.index.isin(session.index).all()
    results, counts = describe(session.join(votes, validate="one_to_one"), visits)
    assert len(results) == 308 and results["p"].between(0, 1).all()
    report(results, counts, order, fixed.first())
    cols = ["Outcome", "Comparison", "Contrast", "Estimate", "95% CI low", "95% CI high", "p", "Holm p", "Order-adjusted estimate"]
    print(results.groupby(["Family", "Comparison"]).size())
    print(results.loc[(results["Holm p"] < .05) | ((results["Outcome"].isin(["PMV", "SET (°C)", "Thermal sensation vote", "Thermal comfort vote", "Tonic EDA", "RMSSD"])) & results["Comparison"].eq("Scenario")), cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
