"""Compare three-hour response models with and without continuous achieved PMV."""
from html import escape
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.stats import norm, t

revised = import_module("06_Revised_Manuscript_Figures")
mixed, trajectory, paired = revised.mixed, revised.trajectory, revised.paired
OUT = revised.OUT / "Achieved PMV model comparison.html"
NODES = (31, 45)


def load():
    _, frames = trajectory.load(clean_eda=True)
    votes, audit = revised.session_votes()
    visits = frames[["part-id", "scenario", "visit", "order"]].drop_duplicates()
    sessions = frames.groupby(paired.KEY).median(numeric_only=True)
    assert len(sessions) == 88 and sessions.pmv.notna().all()
    data = sessions.drop(columns=["visit", "order"]).join(votes).reset_index()
    data = data.merge(visits, on=["part-id", "scenario"], validate="many_to_one")
    assert data[["tsv", "tcv"]].count().eq(83).all()
    return data, audit


def effect(model, weights, convert, critical):
    beta, cov = model
    estimate = weights @ beta
    width = critical * np.sqrt(weights @ cov @ weights)
    return np.asarray(convert(np.array([estimate, estimate-width, estimate+width])))


def response_transform(key, values):
    return np.log(values) if key in paired.LOG else values


def formatted(interval):
    return f"{interval[0]:.3f} [{interval[1]:.3f}, {interval[2]:.3f}]"


def analyze(data):
    rows, slopes, diagnostics, standardized = [], [], [], []
    for family in ("Physiology", "Questionnaire"):
        for key, label in paired.FAMILIES[family].items():
            print(f"Fitting {label}", flush=True)
            d = data.dropna(subset=[key])
            vote = key in paired.ORDINAL
            person = d["part-id"].to_numpy()
            n = d["part-id"].nunique()
            y = response_transform(key, d[key].to_numpy())
            x = np.array([mixed.row(sc, int(period == "Afternoon"), order, not vote)
                          for sc, period, order in zip(d.scenario, d.cluster, d.order)])
            if vote:
                x = x[:, 1:]
            # Center at the nominal target and report the slope per 0.1 PMV.
            designs = (x, np.column_stack([x, (d.pmv.to_numpy() - .5) / .1]))
            models, residual_sds = [], []
            for design, spec in zip(designs, ("Without PMV", "With PMV")):
                assert np.linalg.matrix_rank(design) == design.shape[1]
                args = (design, y, person, d.visit.to_numpy())
                if vote:
                    beta, cov, sd, drift, loss = mixed.ordinal(*args, nodes=NODES)
                else:
                    beta, cov, sd = mixed.gaussian(*args)
                    drift = loss = np.nan
                models.append((beta, cov))
                if not vote:
                    residual_sds.append(sd[2])
                diagnostics.append({"Outcome": label, "Model": spec, "Observations": len(d),
                                    "Participants": n, "Visits": d.visit.nunique(),
                                    "Participant SD": sd[0], "Visit SD": sd[1],
                                    "Residual SD": np.nan if vote else sd[2],
                                    "Ordinal 31 to 45 node coefficient drift": drift,
                                    "Ordinal negative log likelihood": loss})
            scale = ("Odds ratio" if vote else "Tonic EDA ratio" if key == "eda_tonic"
                     else "Ratio" if key in paired.LOG else "Difference in original units")
            convert = np.exp if vote or key in paired.LOG else np.asarray
            critical = norm.ppf(.975) if vote else t.ppf(.975, n-1)
            for kind, contrast, weights in mixed.contrasts(not vote):
                if vote:
                    weights = weights[1:]
                base = effect(models[0], weights, convert, critical)
                adjusted = effect(models[1], np.r_[weights, 0.], convert, critical)
                rows.append({"Outcome": label, "Comparison": kind, "Contrast": contrast,
                             "Scale": scale, "Without PMV": formatted(base),
                             "With PMV": formatted(adjusted)})
                if not vote and kind == "Scenario":
                    model_weights = (weights, np.r_[weights, 0.])
                    intervals = [
                        effect(model, w, np.asarray, critical) / sd
                        for model, w, sd in zip(models, model_weights, residual_sds)
                    ]
                    standardized.append({"Outcome": label, "Contrast": contrast,
                                         "Without PMV": formatted(intervals[0]),
                                         "With PMV": formatted(intervals[1])})
            pmv_term = effect(models[1], np.r_[np.zeros(x.shape[1]), 1.], convert, critical)
            slopes.append({"Outcome": label, "Scale": scale, "Per +0.1 PMV": formatted(pmv_term)})
    return tuple(pd.DataFrame(records) for records in (rows, slopes, diagnostics, standardized))


def main():
    data, audit = load()
    rows, slopes, diagnostics, standardized = analyze(data)
    coverage = data.groupby("scenario").pmv.agg(["size", "min", "median", "max"]).reset_index()
    sections = [f"""<h1>Three-hour response models with and without achieved PMV</h1>
<p>Both models use identical records for each outcome. Physiology and calibrated archived PMV
use session medians from lab minutes 0–179, excluding breaks and pooling identical timestamps.
All 83 recorded endpoint questionnaires contribute to each ordinal model, including the
{int((~audit.eligible).sum())} recorded outside the nominal three-hour clock window. Two questionnaire
scenario labels follow the corresponding logged visit matched by participant, date and time of day.</p>
<p>Both models include categorical scenario and visit order, time of day, and participant and
nested visit random intercepts. Physiology models also include scenario by time of day.
Scenario contrasts average morning and afternoon equally. Ordinal cumulative-logit models
assume proportional odds. A higher TSV means warmer and a higher TCV means more discomfort.
The PMV model adds one common linear slope centered at +0.5. Its scenario contrasts hold PMV
fixed. Tonic EDA, RMSSD and LF/HF use logarithms. One nonpositive tonic EDA reading was excluded before session aggregation.</p>
<p>Estimates are shown with pointwise 95% intervals. Gaussian models use REML with a participant-based
t reference. Ordinal models use likelihood quadrature and approximate Wald intervals. A zero
random-effect SD is a boundary estimate. PMV is partly determined by scenario, so conditional
contrasts describe the fitted association at a common PMV. Comparing the two intervals does
not test the difference between estimates.</p>
<h2>Achieved PMV coverage</h2>""", paired.table(coverage)]
    for label, block in rows.loc[rows.Comparison.eq("Scenario")].groupby("Outcome", sort=False):
        sections.extend([f"<h2>{escape(label)}</h2>", paired.table(block.drop(columns=["Outcome", "Comparison"]))])
    sections.extend(["<h2>Standardized physiology contrasts</h2>",
                     """<p>Model-based Cohen’s d divides the scenario contrast by the fitted residual
SD of the corresponding mixed model, following the original analysis. Tonic EDA, RMSSD and LF/HF use logs. Pointwise 95% intervals rescale the model contrast intervals and
treat the estimated residual SD as fixed. The two specifications have separately estimated
residual SDs.</p>""", paired.table(standardized),
                     "<h2>Achieved PMV slopes</h2>", paired.table(slopes),
                     "<h2>Time of day and visit order effects</h2>",
                     paired.table(rows.loc[~rows.Comparison.eq("Scenario")]),
                     "<h2>Fit diagnostics</h2>", paired.table(diagnostics)])
    style = """body{font:15px/1.5 Arial;color:#182d38;margin:36px auto;max-width:1100px;padding:0 24px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:6px 8px;text-align:left;border-bottom:1px solid #dce5ea}
th{background:#e8f0f4}p{max-width:92ch}h2{margin-top:28px}"""
    OUT.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                   '<title>Achieved PMV model comparison</title>'
                   f'<style>{style}</style><main>{"".join(sections)}</main></html>', encoding="utf-8")
    print(coverage.round(3).to_string(index=False))
    print(rows.loc[rows.Comparison.eq("Scenario")].to_string(index=False))
    print(standardized.to_string(index=False))
    print(slopes.to_string(index=False))
    print(diagnostics.round(5).to_string(index=False))
    print("Endpoint questionnaires outside nominal window:", int((~audit.eligible).sum()))


if __name__ == "__main__":
    main()
