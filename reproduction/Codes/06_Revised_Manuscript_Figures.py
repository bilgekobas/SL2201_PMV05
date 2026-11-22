"""Build the main figures and tables using the nominal three-hour exposure."""
from importlib import import_module
from shutil import copyfile

import pandas as pd

trajectory = import_module("05_AR1_Trajectory_Analysis")
figures = trajectory.figures
mixed = trajectory.mixed
paired = mixed.paired
SOURCE = figures.OUT
OUT = SOURCE / "Revised manuscript"
KEY = paired.KEY
SCENARIOS = paired.CONDITIONS
PERIODS = paired.PERIODS
EXPOSURES = {
    "tdb": "Air temperature (°C)", "tg":"Globe temperature (°C)",
    "mrt": "Mean radiant temperature (°C)", "top": "Operative temperature (°C)",
    "rh": "Relative humidity (%)", "v": "Recorded air speed (m/s)",
    "clo": "Clothing insulation (clo)", "met": "Metabolic rate (met)",
    "co2": "CO₂ (ppm)", "pmv": "PMV", "SET_C": "SET (°C)",
}
MAIN_OUTCOMES = ["PMV", "SET (°C)", *paired.FAMILIES["Physiology"].values(),
                 *paired.FAMILIES["Questionnaire"].values()]


def session_votes():
    source = pd.read_csv(paired.DATA / "SET_timeseries_with_SET.csv")
    source = source.loc[source["lab-id"].eq("SL") & source["location"].eq("Lab")
                        & source["break"].ne(1) & source["time-elapsed"].between(0,179)].copy()
    source["clock"] = pd.to_datetime(source["date"]+" "+source["time"], format="%d-%m-%y %H:%M:%S")
    source["visit_date"] = source.clock.dt.normalize()
    source["origin"] = source.clock - pd.to_timedelta(source["time-elapsed"], unit="min")
    keys = ["part-id", "visit_date", "cluster"]
    assert source.groupby(keys)[["origin","scenario"]].nunique().eq(1).all().all()
    sessions = source[keys+["origin","scenario"]].drop_duplicates()
    votes = pd.read_excel(paired.DATA / "00_SL2201_SubjectiveResponses.xlsx", header=1)
    votes = votes.dropna(subset=["part-id","scenario","cluster","datetime"])[KEY+["datetime","tsv","tcv"]].copy()
    votes["clock"] = pd.to_datetime(votes.pop("datetime"))
    votes["visit_date"] = votes.clock.dt.normalize()
    votes["part-id"] = "SL2201_" + votes["part-id"].str.strip()
    votes["cluster"] = votes["cluster"].str.split("_").str[-1]
    votes = votes.rename(columns={"scenario":"questionnaire_scenario"}).merge(sessions, on=keys, validate="one_to_one", how="left")
    assert len(votes)==83 and votes.origin.notna().all() and votes[["tsv","tcv"]].notna().all().all()
    votes["eligible"] = (votes.clock-votes.origin).dt.total_seconds().between(0,10800,inclusive="left")
    label_agrees = votes.scenario.eq(votes.questionnaire_scenario)
    ambiguous = votes.loc[~label_agrees, "eligible"]
    assert len(ambiguous)==2 and ambiguous.sum()==1
    # Use the logged scenario for the visit matched by participant, date and time of day.
    retained = votes.set_index(KEY)[["tsv","tcv"]]
    assert len(retained)==83 and retained.index.is_unique
    return retained, votes


def exposure_summary(session):
    rows = []
    for key, label in EXPOSURES.items():
        out = {"Measure":label}
        for sc in SCENARIOS:
            values = session.xs(sc,level="scenario")[key]
            q1, median, q3 = values.quantile([.25,.5,.75])
            out[sc] = f"{median:.3g} [{q1:.3g}, {q3:.3g}]" if values.nunique()>1 else f"{median:.3g}"
        rows.append(out)
    return pd.DataFrame(rows)


def figure_copy(session):
    figures.OUT = OUT
    figures.draw_exposure(session, window_note="Medians use in-lab minutes 0–179. Breaks are excluded.")
    figures.draw_absolute_responses(session,
        window_note="Physiology shows in-lab medians for minutes 0–179 (n=11 per cell). Breaks are excluded.",
        vote_note="Votes are session endpoints. Gaps indicate unavailable votes. Higher TCV means more discomfort.")
    figures.draw_responses(session,
        window_note="Physiology uses minutes 0–179. Each dot is one participant. Votes use available session endpoints.")
    (OUT / "Supplement - Paired responses versus T1.svg").replace(OUT / "Figure S2 - Paired responses versus T1.svg")
    copyfile(SOURCE / "AR1 climate and physiology trajectories.svg",
             OUT / "AR1 climate and physiology trajectories.svg")
    copyfile(SOURCE / "Trajectory analysis.html", OUT / "Trajectory analysis.html")


def counts_table(votes):
    vote_rows = []
    for sc in SCENARIOS:
        for period in PERIODS:
            cell = votes.loc[votes.scenario.eq(sc) & votes.cluster.eq(period)]
            n = len(cell)
            in_window = int(cell.eligible.sum())
            conflicts = int(cell.scenario.ne(cell.questionnaire_scenario).sum())
            vote_rows.append({"Scenario":sc,"Time":period,"Sessions":11,
                              "No vote":11-n,"Outside 3 h":n-in_window,
                              "Visit scenario used":conflicts,"Included TSV/TCV":n})
    return pd.DataFrame(vote_rows)


def report(session, votes, results, diagnostics):
    cols = ["Outcome","Contrast","Model scale","Model estimate","95% CI low","95% CI high","Model Holm p"]
    headline = results.loc[results["Outcome"].isin(MAIN_OUTCOMES) & results.Comparison.eq("Scenario"),cols]
    sections = ["""<h1>Three-hour manuscript figures and tables</h1>
<p>All sensor summaries use the nominal in-lab exposure window of elapsed minutes 0–179, excluding breaks. All recorded questionnaires are included as session endpoints, including those submitted after the three-hour sensor window. Questionnaires are matched on participant, visit date and time of day. This package replaces the earlier figures for the revised manuscript.</p>
<h2>Main figures</h2>
<ol><li><a href="Figure%201%20-%20Achieved%20thermal%20exposure.svg">Achieved PMV and SET</a>. Each dot is one of 88 participant-session medians. Lines show the median and interquartile range for each scenario. The dashed line shows the PMV design target of +0.5.</li>
<li><a href="Figure%202%20-%20Absolute%20physiological%20and%20comfort%20responses.svg">Paired physiology, sensation and comfort</a>. Lines join observations from the same participant within morning or afternoon. Black bars show the median in each cell. Sensor points are session medians and vote points are recorded responses. Higher TSV means warmer and higher TCV means more discomfort.</li></ol>
<h2>Table 1 · Achieved environmental conditions</h2>
<p>Median [first quartile, third quartile] of 22 participant-session medians in each scenario. A single number indicates a constant across those sessions. Recorded air speed is constant within each scenario in the saved file.</p>""",
                paired.table(exposure_summary(session)),
                """<h2>Table 2 · Adjusted scenario contrasts</h2>
<p>Continuous outcomes use Gaussian mixed models with scenario, time of day, scenario × time of day and categorical visit order. Random intercepts account for participant and visit nested within participant. Estimates average morning and afternoon equally. Positive differences indicate higher values in the scenario named first. Ratios above 1 indicate higher geometric levels. Tonic EDA uses a plain log transform after excluding one negative reading. TSV and TCV use additive cumulative-logit mixed models with scenario, time of day, and categorical visit order and the same random intercepts. Odds ratios above 1 indicate higher vote categories. Time, visit and exploratory interaction contrasts are in the supplementary tables. The 95% intervals are unadjusted. Holm p values adjust across contrasts within outcome family and comparison type.</p>""",
                paired.table(headline),
                """<h2>Supplementary figures and tables</h2>
<ul><li>Figure S1. <a href="AR1%20climate%20and%20physiology%20trajectories.svg">Unbinned AR(1) trajectories</a>. <a href="Trajectory%20analysis.html">Full trajectory results and same-window median comparison</a>.</li>
<li>Figure S2. <a href="Figure%20S2%20-%20Paired%20responses%20versus%20T1.svg">Within-participant differences versus T1</a>. Morning and afternoon differences are averaged when both are available.</li></ul>
<h3>Table S1 · Vote availability</h3>
<p>Of 88 sessions, 83 had a questionnaire response. All 83 are included, including 35 recorded at or after minute 180. Two questionnaire scenario labels disagreed with the logged visit. Both responses are assigned to the scenario recorded for the matching participant, date and time of day. The questionnaire estimates describe session endpoints, with variation in response timing.</p>""",
                paired.table(counts_table(votes)),
                "<h3>Table S2 · Session-model diagnostics</h3>",
                paired.table(diagnostics)]
    for (family, kind), block in results.groupby(["Family","Comparison"],sort=False):
        sections += [f"<h3>{family} · {kind}</h3>",paired.table(block.drop(columns=["Family","Comparison"]))]
    style = """body{font:15px/1.5 Arial;color:#182d38;margin:36px auto;max-width:1320px;padding:0 24px}
p,li{max-width:100ch}h2{margin-top:42px}h3{margin-top:30px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px 8px;text-align:right;border-bottom:1px solid #dce5ea}
td:first-child,th:first-child{text-align:left}th{background:#e8f0f4}tbody tr:nth-child(even){background:#f5f8fa}
a{color:#176a91}"""
    (OUT/"Manuscript figures and tables.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Three-hour manuscript figures and tables</title>'
        f"<style>{style}</style><main>{''.join(sections)}</main></html>",encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw, frame = trajectory.load(clean_eda=True)
    measures = frame.drop(columns=["visit","order","unix","time-elapsed","session"])
    medians = measures.groupby(KEY).median(numeric_only=True).join(
        raw.groupby(KEY)[["v","clo","met"]].median())
    votes, audit = session_votes()
    assert votes.index.isin(medians.index).all() and len(medians)==88
    session = medians.join(votes)
    observed = session
    figure_copy(observed)
    visits = raw[["part-id","scenario","visit","order"]].drop_duplicates()
    results, diagnostics = mixed.analyze(session, visits, ordinal_nodes=(31, 45))
    results.loc[results.Outcome.eq("Tonic EDA"),"Model scale"] = "Tonic EDA ratio"
    report(observed,audit,results,diagnostics)
    print(counts_table(audit).to_string(index=False))
    print(results.loc[results.Outcome.isin(["PMV","SET (°C)","RMSSD","Thermal sensation vote","Thermal comfort vote"]) &
                      (results["Model Holm p"].lt(.05) | results["Contrast"].eq("T4 − T2"))]
          .round(4).to_string(index=False))
    print(diagnostics.tail(2).to_string(index=False))


if __name__ == "__main__":
    main()
