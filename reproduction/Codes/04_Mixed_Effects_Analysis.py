"""Fit session-level Gaussian and cumulative-logit mixed models."""
from html import escape
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm
from scipy.stats import norm, t

from importlib import import_module
paired = import_module("03_Repeated_Measures_Analysis")

CONDITIONS = paired.CONDITIONS
KEY = paired.KEY
FAMILIES = paired.FAMILIES
OUT = paired.OUT.with_name("Mixed-effects analysis.html")


def gaussian(x, y, participant, visit):
    """Profiled REML for residual, participant and nested visit variance."""
    n, p = x.shape
    i = np.eye(n)
    person = (participant[:, None] == participant).astype(float)
    day = (visit[:, None] == visit).astype(float)
    y = (y - (center := y.mean())) / (scale := y.std())
    def profile(ratios):
        chol = cho_factor(i + ratios[0] * person + ratios[1] * day)
        vx = cho_solve(chol, x)
        info = x.T @ vx
        beta = np.linalg.solve(info, vx.T @ y)
        resid = y - x @ beta
        residual_var = resid @ cho_solve(chol, resid) / (n - p)
        criterion = ((n-p) * np.log(residual_var) + 2 * np.log(np.diag(chol[0])).sum()
                     + np.linalg.slogdet(info)[1])
        return criterion, beta, residual_var, info

    def objective(ratios):
        return profile(ratios)[0]

    fits = [minimize(objective, start, method="L-BFGS-B", bounds=[(0, 1e4)] * 2)
            for start in ([1., 1.], [0., 0.], [10., .2])]
    fit = min((f for f in fits if f.success), key=lambda f: f.fun)
    _, beta, resvar, info = profile(fit.x)
    beta = beta * scale
    beta[0] += center
    cov = np.linalg.inv(info) * resvar * scale**2
    return beta, cov, np.sqrt(resvar * scale**2 * np.r_[fit.x, 1.])


def hessian(fun, x, h=2e-4):
    eye = np.eye(len(x)) * h
    out = np.empty((len(x), len(x)))
    base = fun(x)
    for i, a in enumerate(eye):
        out[i, i] = (fun(x + a) - 2 * base + fun(x - a)) / h**2
        for j, b in enumerate(eye[:i]):
            out[i, j] = out[j, i] = (fun(x+a+b)-fun(x+a-b)-fun(x-a+b)+fun(x-a-b))/(4*h*h)
    return out


def ordinal(x, y, participant, visit, nodes=(31, 45)):
    """Marginal cumulative-logit likelihood with nested Gaussian intercepts."""
    categories, codes = np.unique(y, return_inverse=True)
    k, p = len(categories) - 1, x.shape[1]
    _, participant_idx = np.unique(participant, return_inverse=True)
    _, visit_idx = np.unique(visit, return_inverse=True)
    people = participant_idx.max() + 1
    visits = visit_idx.max() + 1
    visit_person = np.empty(visits, dtype=int)
    visit_person[visit_idx] = participant_idx

    def likelihood(nodes):
        z, w = roots_hermitenorm(nodes)
        logw = np.log(w / np.sqrt(2 * np.pi))
        def objective(params):
            beta, cut = params[:p], params[p:p+k]
            thresholds = np.r_[-np.inf, cut[0], cut[0]+np.cumsum(np.exp(cut[1:])), np.inf]
            eta = (x @ beta)[:, None, None] + params[-2] * z[None, :, None] + params[-1] * z[None, None, :]
            upper = expit(thresholds[codes+1, None, None] - eta)
            lower = expit(thresholds[codes, None, None] - eta)
            by_visit = np.zeros((visits, nodes, nodes))
            np.add.at(by_visit, visit_idx, np.log(np.clip(upper-lower, 1e-100, 1)))
            by_person = np.zeros((people, nodes))
            np.add.at(by_person, visit_person, logsumexp(by_visit+logw, axis=2))
            return -logsumexp(by_person+logw, axis=1).sum()
        return objective

    cum = np.bincount(codes).cumsum()[:-1] / len(y)
    logits = np.log(cum / (1-cum))
    start = np.r_[np.zeros(p), logits[0], np.log(np.diff(logits)), .6, .4]
    bounds = [(None, None)] * (p+1) + [(-10, 5)] * (k-1) + [(0, 7)] * 2
    coarse = likelihood(15)
    fits = [minimize(coarse, np.r_[start[:-2], sd], bounds=bounds,
                     method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-6})
            for sd in ([.6,.4], [2.,.6], [.6,2.])]
    best = min(fits, key=lambda f: f.fun)
    objective = likelihood(nodes[0])
    fit = minimize(objective, best.x, bounds=bounds, method="L-BFGS-B",
                   options={"ftol": 1e-12, "gtol": 1e-6})
    fine = likelihood(nodes[1])
    final = minimize(fine, fit.x, bounds=bounds, method="L-BFGS-B",
                     options={"ftol": 1e-12, "gtol": 1e-6})
    assert final.success and np.max(final.x[-2:]) < 6.9, final.message
    # A variance component at zero is fixed there for local Wald intervals.
    active = np.r_[np.arange(p+k), p+k+np.flatnonzero(final.x[-2:] > 1e-4)]
    def reduced(theta):
        full = final.x.copy()
        full[active] = theta
        return fine(full)
    info = hessian(reduced, final.x[active])
    assert np.linalg.eigvalsh(info).min() > 0
    return (final.x[:p], np.linalg.inv(info)[:p,:p], final.x[-2:],
            np.max(np.abs(fit.x[:p]-final.x[:p])), final.fun)


def row(scenario, after, visit, interaction):
    indicators = np.array([scenario == sc for sc in CONDITIONS[1:]], dtype=float)
    base = [1, *indicators, after, *[visit == order for order in (2,3,4)]]
    return np.array(base + (list(after * indicators) if interaction else []), dtype=float)


def contrasts(interaction):
    def time(sc):
        return row(sc, 1, 1, interaction) - row(sc, 0, 1, interaction)

    for a, b in combinations(CONDITIONS, 2):
        avg = sum(row(b, period, 1, interaction) - row(a, period, 1, interaction) for period in (0,1)) / 2
        yield "Scenario", f"{b} − {a}", avg
        if interaction:
            yield "Scenario by time", f"{b} − {a}", time(b) - time(a)
    if interaction:
        for sc in CONDITIONS:
            yield "Time within scenario", sc, time(sc)
    else:
        yield "Time across scenarios", "Afternoon − Morning", time("T1")
    for order in (2,3,4):
        yield "Visit order", f"{order} − 1", row("T1", 0, order, interaction) - row("T1", 0, 1, interaction)


def read_data():
    data = pd.read_csv(paired.DATA / "SET_timeseries_with_SET.csv")
    data = data.loc[data["lab-id"].eq("SL") & data["location"].eq("Lab") & data["break"].ne(1)]
    visit_info = data[["part-id", "scenario", "date"]].drop_duplicates().copy()
    assert len(visit_info) == 44
    visit_info["order"] = pd.to_datetime(visit_info.date, format="%d-%m-%y").groupby(visit_info["part-id"]).rank()
    visit_info["visit"] = np.arange(len(visit_info))
    votes = pd.read_excel(paired.DATA / "00_SL2201_SubjectiveResponses.xlsx", header=1)
    votes = votes[KEY + ["tsv", "tcv"]].dropna(subset=KEY).copy()
    votes["part-id"] = "SL2201_" + votes["part-id"].str.strip()
    votes["cluster"] = votes["cluster"].str.split("_").str[-1]
    session = data.groupby(KEY).median(numeric_only=True)
    votes = votes.set_index(KEY)
    assert len(session) == 88 and len(votes) == 83 and votes.index.is_unique and votes.index.isin(session.index).all()
    session = session.join(votes)
    return session, visit_info


def analyze(session, visits, ordinal_nodes=(31, 45)):
    data = session.reset_index().merge(visits, on=["part-id","scenario"], validate="many_to_one")
    rows, diagnostics = [], []
    for family, outcomes in FAMILIES.items():
        for key, name in outcomes.items():
            d = data.dropna(subset=[key])
            y = d[key].to_numpy()
            is_vote = key in paired.ORDINAL
            is_log = key in paired.LOG
            x = np.array([row(sc, int(period == "Afternoon"), v, not is_vote)
                          for sc, period, v in zip(d.scenario, d.cluster, d.order)])
            if is_vote:
                x = x[:,1:]
            assert np.linalg.matrix_rank(x) == x.shape[1]
            args = (x, np.log(y) if is_log else y, d["part-id"].to_numpy(), d.visit.to_numpy())
            if is_vote:
                beta, cov, sd, drift, loss = ordinal(*args, nodes=ordinal_nodes)
            else:
                beta, cov, sd = gaussian(*args)
                drift = loss = np.nan
            diagnostics.append({"Outcome": name, "Sessions": len(d), "Participants": d["part-id"].nunique(),
                                "Participant SD": sd[0], "Nested visit SD": sd[1],
                                "Residual SD": np.nan if is_vote else sd[2],
                                f"Max coefficient shift {ordinal_nodes[0]}→{ordinal_nodes[1]} nodes": drift, "Ordinal negative log likelihood": loss})
            transform = np.exp if is_vote or is_log else float
            reference = norm if is_vote else t(df=d["part-id"].nunique()-1)
            for kind, contrast, weights in contrasts(not is_vote):
                if is_vote:
                    weights = weights[1:]
                estimate = weights @ beta
                se = np.sqrt(weights @ cov @ weights)
                half = reference.ppf(.975) * se
                rows.append({"Family": family,"Outcome": name,"Comparison": kind,"Contrast": contrast,
                             "Model scale": "Odds ratio" if is_vote else "Ratio" if is_log else "Difference",
                             "Model estimate": transform(estimate), "95% CI low": transform(estimate-half),
                             "95% CI high": transform(estimate+half), "Model p": 2*reference.sf(abs(estimate/se))})
    results = pd.DataFrame(rows)
    results["Model Holm p"] = results.groupby(["Family","Comparison"])["Model p"].transform(paired.holm)
    return results, pd.DataFrame(diagnostics)


def main():
    session, visits = read_data()
    model, diagnostics = analyze(session, visits)
    original, _ = paired.describe(session, visits.pivot(index="part-id",columns="scenario",values="order"))
    # Compare the additive ordinal time estimate with a pooled paired direction.
    pooled = []
    for key, label in FAMILIES["Questionnaire"].items():
        by_period = session[key].unstack("cluster")
        direction = np.sign(by_period["Afternoon"] - by_period["Morning"])
        values = direction.groupby(level="part-id").mean().dropna()
        pooled.append(paired.record("Questionnaire", key, label, "Time across scenarios",
                                    "Afternoon − Morning", values, len(values)))
    pooled = pd.DataFrame(pooled)
    pooled["Holm p"] = paired.holm(pooled["p"])
    original = pd.concat([original, pooled], ignore_index=True)
    keys = ["Family","Outcome","Comparison","Contrast"]
    comparison = model.merge(original[keys+["Scale","Estimate","Holm p"]], on=keys, how="left", validate="one_to_one")
    comparison = comparison.rename(columns={"Scale":"Paired scale","Estimate":"Paired estimate","Holm p":"Paired Holm p"})
    print(diagnostics.to_string(index=False))
    selected = comparison["Model Holm p"].lt(.05) | (comparison["Comparison"].eq("Scenario") &
        comparison["Outcome"].isin(["PMV","SET (°C)","Tonic EDA","Thermal sensation vote","Thermal comfort vote"]))
    selected |= comparison["Outcome"].eq("Heart rate (bpm)") & comparison["Comparison"].eq("Time within scenario")
    print(comparison.loc[selected].round(4).to_string(index=False))
    sections = ["""<h1>Mixed-effects analysis</h1>
<p>88 sessions from 44 visits and 11 participants. Continuous outcomes use in-lab session medians with breaks excluded. The five missing questionnaire sessions were omitted. Air speed, clothing and metabolic rate were recorded as constants within scenario and cannot support fitted within-scenario variation.</p>
<h2>Model specifications</h2>
<p>Continuous outcomes use Gaussian restricted maximum likelihood with fixed scenario, time of day, scenario × time of day, and categorical visit order (visits 1–4). The model includes random intercepts for participants and visits nested in participants. EDA, RMSSD and LF/HF are fitted on the log scale and reported as ratios. Scenario contrasts average morning and afternoon equally. The fixed visit effect adjusts for a common order pattern. It does not establish complete control of day or carryover effects. Intervals and tests use plug-in covariance and a t reference with 10 degrees of freedom as a small-sample approximation.</p>
<p>TSV and TCV use proportional-odds cumulative-logit mixed models with additive scenario, time of day and categorical visit order, and the same two levels of random intercept. The likelihood integrates the random intercepts using Gaussian quadrature. Odds ratios above 1 indicate higher votes, meaning warmer for TSV and more uncomfortable for TCV. Their Wald intervals and p values are approximate with 11 participants and sparse extreme categories. The common odds ratio assumes proportional odds. The prior paired vote analysis instead estimates a net probability of a higher versus lower vote, so compare directions rather than magnitudes.</p>
<p>Holm adjustment is applied within previous outcome families and comparison types. Intervals are unadjusted. For scenario contrasts the paired analysis is unadjusted for visit order. The paired comparison for additive ordinal time averages the direction of the available afternoon–morning differences within participant. Fixed visit-order contrasts have no paired counterpart. Random-effect SDs of zero indicate boundary estimates. These results are exploratory.</p>
<h2>Fit diagnostics</h2>""", paired.table(diagnostics)]
    for (family, kind), block in comparison.groupby(["Family","Comparison"], sort=False):
        sections.extend([f"<h2>{escape(family)} · {escape(kind)}</h2>",
                         paired.table(block.drop(columns=["Family","Comparison"]))])
    style = """body{font:15px/1.45 Arial;color:#182d38;margin:36px auto;max-width:1420px;padding:0 22px}
p{max-width:98ch}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:6px 8px;text-align:right;border-bottom:1px solid #dce5ea}
td:first-child,th:first-child{text-align:left}th{background:#e8f0f4}tr:nth-child(even){background:#f5f8fa}"""
    OUT.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Mixed-effects analysis</title>'
                   f"<style>{style}</style><main>{''.join(sections)}</main></html>", encoding="utf-8")


if __name__ == "__main__":
    main()
