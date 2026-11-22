"""Fit unbinned trajectories with continuous-time AR(1) within sessions."""
from importlib import import_module
from itertools import combinations
from scipy.linalg import cholesky, solve as linear_solve

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t

mixed = import_module("04_Mixed_Effects_Analysis")
paired = mixed.paired
figures = import_module("02_Rebuild_Results_Figures")
KEY = paired.KEY
SCENARIOS = paired.CONDITIONS
FAMILIES = {k: v for k, v in paired.FAMILIES.items() if k != "Questionnaire"}
OUT = paired.OUT.with_name("Trajectory analysis.html")
EXP = {"eda_tonic", "rmssd", "lf/hf"}
COLORS = ["#455b70", "#a85927", "#18788a", "#8c4d79"]


def transform(key, y):
    if key in paired.LOG:
        return np.log(y)
    return y


def design(sc, period, order, minute):
    # Midpoint is the mean prediction on the nominal minute 0–179 grid.
    elapsed = (minute - 89.5) / 60
    return np.r_[mixed.row(sc, period, order, True), elapsed,
                 *[elapsed * (sc == s) for s in SCENARIOS[1:]], elapsed * period]


def block_stats(d, x, y):
    """Form sufficient statistics by lag for the AR innovations likelihood."""
    z = np.column_stack([np.ones(len(d)),
                         pd.get_dummies(d["scenario"], dtype=float).to_numpy(),
                         pd.get_dummies(d["session"], dtype=float).to_numpy()])
    w = np.column_stack([x, y, z])
    delta = d.groupby("session").unix.diff().to_numpy() / 60
    first = np.isnan(delta)
    initial = w[first].T @ w[first]
    gaps = np.unique(delta[~first])
    assert np.all(gaps > 0)
    moments, counts = [], []
    for gap in gaps:
        pos = np.flatnonzero(delta == gap)
        current, previous = w[pos], w[pos-1]
        cross = current.T @ previous
        moments.append([current.T @ current, cross + cross.T, previous.T @ previous])
        counts.append(len(pos))
    return initial, gaps, np.asarray(moments), np.asarray(counts), [1,4,8]


def corrected_gram(block, rho, ratios):
    initial, gaps, moments, counts, sizes = block
    a = rho ** gaps
    innovation = 1-a*a
    coefficients = np.column_stack([np.ones(len(a)), -a, a*a]) / innovation[:,None]
    gram = initial + np.einsum("gl,glij->ij", coefficients, moments)
    z_dim = sum(sizes)
    root = np.sqrt(np.repeat(ratios, sizes))
    zz = np.eye(z_dim) + gram[-z_dim:, -z_dim:] * np.outer(root, root)
    wz = gram[:-z_dim, -z_dim:] * root
    adjusted = gram[:-z_dim,:-z_dim] - wz @ linear_solve(zz,wz.T,assume_a="pos")
    return adjusted, counts @ np.log(innovation) + (2*np.log(np.diag(cholesky(zz))).sum())


def trajectory(d, x, y):
    """Profiled REML, followed by a participant deletion covariance estimate."""
    n, p = x.shape
    center, scale = y.mean(), y.std()
    ys = (y-center) / scale
    blocks = [block_stats(d.iloc[idx], x[idx], ys[idx])
              for idx in d.groupby("part-id", sort=False).indices.values()]

    def solve(params, details=False):
        grams, determinants = zip(*(corrected_gram(b, params[0], params[1:]) for b in blocks))
        total = np.sum(grams, axis=0)
        information, xy = total[:-1,:-1], total[:-1,-1]
        beta = linear_solve(information,xy,assume_a="pos")
        var = max((total[-1,-1] - xy @ beta) / (n-p), 1e-15)
        criterion = (n-p)*np.log(var) + sum(determinants) + (2*np.log(np.diag(cholesky(information))).sum())
        if details:
            return criterion, beta, grams, total, var
        return criterion

    def unpack(params):
        return np.r_[np.exp(-np.exp(params[0])), np.expm1(params[1:])]

    def objective(params):
        return solve(unpack(params))

    bounds = [(np.log(-np.log(.99999)), np.log(-np.log(.000001)))]+[(0,np.log1p(1e5))]*3
    fits = [minimize(objective, np.r_[np.log(-np.log(s[0])),np.log1p(s[1:])],
                     method="L-BFGS-B",bounds=bounds,options={"ftol":1e-11,"maxiter":350})
            for s in ([.8,.1,.1,.1],[.98,2.,1.,.5],[.999,2.,2.,2.])]
    valid = [fit for fit in fits if fit.success]
    if not valid:
        raise RuntimeError([fit.message for fit in fits])
    fit = min(valid,key=lambda f:f.fun)
    params = unpack(fit.x)
    _, beta, grams, total, residual_var = solve(params, True)
    deleted = []
    for gram in grams:
        kept = total-gram
        deleted.append(linear_solve(kept[:-1,:-1],kept[:-1,-1],assume_a="pos"))
    estimates = np.asarray(deleted)
    deviations = estimates - estimates.mean(axis=0)
    groups = len(blocks)
    cov = (groups-1)/groups * deviations.T @ deviations * scale**2
    beta = beta*scale
    beta[0] += center
    diagnostics = {"One-minute AR correlation":params[0],
                   "Participant SD":np.sqrt(params[1]*residual_var)*scale,
                   "Visit SD":np.sqrt(params[2]*residual_var)*scale,
                   "Session SD":np.sqrt(params[3]*residual_var)*scale,
                   "Residual SD":np.sqrt(residual_var)*scale,
                   "REML objective gap across starts":max(f.fun for f in valid)-fit.fun,
                   "Optimized starts":len(valid)}
    return beta,cov,diagnostics


def verify_whitening():
    # Use the real design dimensions in a single-participant subset.
    d = pd.DataFrame({"unix":np.tile([0,60,180],8), "session":np.repeat(np.arange(8),3),
                      "scenario":np.repeat(np.arange(4),6)})
    x = np.column_stack([np.ones(len(d)), np.linspace(-1,1,len(d))])
    y = np.sin(np.arange(len(d)))
    rho = .8
    var = [.4,.2,.3]
    gram, ld = corrected_gram(block_stats(d,x,y),rho,var)
    same_session = np.equal.outer(d.session.to_numpy(),d.session.to_numpy())
    same_visit = np.equal.outer(d.scenario.to_numpy(),d.scenario.to_numpy())
    gap = np.abs(np.subtract.outer(d.unix.to_numpy(),d.unix.to_numpy())) / 60
    dense = np.where(same_session,rho**gap,0) + var[0] + var[1]*same_visit + var[2]*same_session
    values = np.column_stack([x,y])
    np.testing.assert_allclose(gram,values.T @ linear_solve(dense,values,assume_a="pos"),atol=1e-10)
    np.testing.assert_allclose(ld,(2*np.log(np.diag(cholesky(dense))).sum()),atol=1e-10)


def load(clean_eda=False):
    raw = pd.read_csv(paired.DATA / "SET_timeseries_with_SET.csv")
    raw = raw.query("`lab-id` == 'SL' and location == 'Lab' and `break` != 1").copy()
    raw = raw.loc[raw["break"].ne(1) & raw["time-elapsed"].between(0,179)]
    if clean_eda:
        raw.loc[raw.eda_tonic.le(0), "eda_tonic"] = np.nan
    visits = raw[["part-id","scenario","date"]].drop_duplicates().copy()
    visits["order"] = pd.to_datetime(visits.date,format="%d-%m-%y").groupby(visits["part-id"]).rank()
    visits["visit"] = np.arange(len(visits))
    raw = raw.merge(visits,on=["part-id","scenario","date"],validate="many_to_one")
    continuous = [k for outcomes in FAMILIES.values() for k in outcomes]
    index = KEY+["visit","order","unix"]
    # Identical timestamps have the same AR state. Pool such records before fitting.
    frames = raw.groupby(index,as_index=False)[continuous+["time-elapsed"]].median()
    frames["session"] = frames.groupby(KEY).ngroup()
    assert frames.session.nunique()==88 and frames.visit.nunique()==44
    return raw, frames


def average_row(sc, minute):
    return np.mean([design(sc,period,order,minute)
                    for period in (0,1) for order in (1,2,3,4)],axis=0)


def effect(beta,cov,contrast,key):
    est = float(contrast @ beta)
    se = float(np.sqrt(contrast @ cov @ contrast))
    dist = t(10)
    width = dist.ppf(.975)*se
    convert = np.exp if key in EXP else float
    return convert(est),convert(est-width),convert(est+width),2*dist.sf(abs(est/se))


def fit_all(frames):
    stats, comparisons, curves, coverage = [],[],{},[]
    for family,outcomes in FAMILIES.items():
        for key,name in outcomes.items():
            d = frames.dropna(subset=[key]).sort_values(KEY+["unix"]).reset_index(drop=True)
            x = np.array([design(sc,int(p=="Afternoon"),order,minute)
                          for sc,p,order,minute in zip(d.scenario,d.cluster,d.order,d["time-elapsed"])])
            assert np.linalg.matrix_rank(x) == x.shape[1]
            beta,cov,diagnostics = trajectory(d,x,transform(key,d[key].to_numpy()))
            stats.append({"Outcome":name,"Rows":len(d),"Sessions":d.session.nunique(),**diagnostics})
            print(name,len(d),round(diagnostics["One-minute AR correlation"],4),
                  round(diagnostics["REML objective gap across starts"],4),flush=True)
            medians = d.groupby(KEY+["visit","order"],as_index=False)[key].median()
            mx = np.array([mixed.row(sc,int(p=="Afternoon"),order,True)
                           for sc,p,order in zip(medians.scenario,medians.cluster,medians.order)])
            mb, mc, _ = mixed.gaussian(mx,transform(key,medians[key].to_numpy()),
                                       medians["part-id"].to_numpy(),medians.visit.to_numpy())
            for a,b in combinations(SCENARIOS,2):
                label = f"{b} − {a}"
                mid = average_row(b,89.5)-average_row(a,89.5)
                slope = (average_row(b,120)-average_row(b,60))-(average_row(a,120)-average_row(a,60))
                for kind,c in (("Three-hour average",mid),("Difference in slope per hour",slope)):
                    value,lo,hi,p = effect(beta,cov,c,key)
                    result = {"Family":family,"Outcome":name,"Comparison":kind,"Contrast":label,
                              "Scale":"Tonic EDA ratio" if key=="eda_tonic" else "Ratio" if key in EXP else "Difference",
                              "Trajectory estimate":value,"95% CI low":lo,"95% CI high":hi,"p":p}
                    if kind=="Three-hour average":
                        val,mlo,mhi,mp = effect(mb,mc,mid[:len(mb)],key)
                        result["Same-window median"] = val
                        result["Same-window median 95% CI"] = f"{mlo:.3g} to {mhi:.3g}"
                        result["Same-window median p"] = mp
                    comparisons.append(result)
            for minute in (0,15,30,60,90,120,150,165,179):
                nearby = d.loc[(d["time-elapsed"]-minute).abs()<=5]
                coverage.append({"Outcome":name,"Minute ±5":minute,**{
                    s:nearby.loc[nearby.scenario.eq(s),"session"].nunique() for s in SCENARIOS}})
            curves[key] = (beta,cov)
    results = pd.DataFrame(comparisons)
    results["Holm p"] = results.groupby(["Family","Comparison"])["p"].transform(paired.holm)
    mask = results["Comparison"].eq("Three-hour average")
    results.loc[mask,"Same-window median Holm p"] = results.loc[mask].groupby("Family")["Same-window median p"].transform(paired.holm)
    return results, pd.DataFrame(stats), pd.DataFrame(coverage), curves


def plot(curves):
    keys = ["pmv","SET_C","skin-temp-wrist-smooth","eda_tonic","hr-clean","rmssd","lf/hf"]
    labels = ["PMV","SET (°C)","Wrist skin temperature (°C)","Tonic EDA (µS)","Heart rate (bpm)","RMSSD (ms)","LF/HF"]
    fig = figures.Figure(1080,1060)
    fig.text(34,44,"Climate and physiology trajectories",27,weight=700)
    fig.text(34,70,"AR(1) mixed-model curves averaged equally across morning, afternoon and visit order.",color=figures.MUTED)
    for i,(scenario,color) in enumerate(zip(SCENARIOS,COLORS)):
        x = 36+85*i
        fig.line(x,101,x+22,101,color,3)
        fig.text(x+29,106,scenario)
    for i,(key,label) in enumerate(zip(keys,labels)):
        beta,cov = curves[key]
        left,top = 78+510*(i%2),146+215*(i//2)
        minutes = np.arange(0,180)
        paths = []
        for sc in SCENARIOS:
            x = np.array([average_row(sc,m) for m in minutes])
            y = x@beta
            w = t.ppf(.975,10)*np.sqrt(np.einsum("ij,jk,ik->i",x,cov,x).clip(0))
            levels = np.array([y,y-w,y+w])
            if key in EXP:
                levels = np.exp(levels)
            paths.append(levels)
        low,high = min(p.min() for p in paths),max(p.max() for p in paths)
        if key=="pmv":
            low,high = min(low,.5),max(high,.5)
        step0 = 10 ** np.floor(np.log10((high-low)/4))
        step = next(n*step0 for n in (1,2,2.5,5,10) if n*step0>=(high-low)/4)
        lower,upper = np.floor(low/step)*step,np.ceil(high/step)*step
        xp = left+minutes/179*405
        def ypos(value):
            return top+150-(value-lower)/(upper-lower)*128
        fig.text(left-33,top,label,16,weight=700)
        for level in np.arange(lower,upper+step/2,step):
            fig.line(left,ypos(level),left+405,ypos(level),"#dfe7eb")
            fig.text(left-8,ypos(level)+4,f"{0 if abs(level)<1e-10 else level:g}",11,anchor="end")
        for value in (0,30,60,90,120,150,179):
            x = left+value/179*405
            fig.text(x,top+170,f"{value}",11,anchor="middle")
        fig.text(left+202,top+190,"Elapsed minutes",11,anchor="middle")
        if key=="pmv":
            fig.line(left,ypos(.5),left+405,ypos(.5),"#71838b",1,"4 4")
        for color,(curve,lo,hi) in zip(COLORS,paths):
            vertices = list(zip(xp,ypos(lo)))+list(zip(xp[::-1],ypos(hi)[::-1]))
            polygon = " ".join(f"{x:.2f},{y:.2f}" for x,y in vertices)
            fig.svg.append(f'<polygon points="{polygon}" fill="{color}" fill-opacity=".11"/>')
        for color,(curve,_,_) in zip(COLORS,paths):
            points = " ".join(f"{x:.2f},{y:.2f}" for x,y in zip(xp,ypos(curve)))
            fig.svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>')
    fig.text(598,812,"Pointwise 95% intervals",16,weight=700)
    fig.text(598,837,"Uncertainty uses participant deletion.",13,color=figures.MUTED)
    fig.text(598,862,"The dashed line marks the PMV design target.",13,color=figures.MUTED)
    fig.text(598,887,"Curves represent linear trends over minutes 0–179.",13,color=figures.MUTED)
    fig.text(34,1035,"Intervals describe the fitted curves. Consult the coverage table for observed data near each minute.",12,color=figures.MUTED)
    fig.save("AR1 climate and physiology trajectories.svg")


def report(results, diagnostics, coverage, nraw, nunique):
    intro = f"""<h1>Climate and physiology trajectories</h1>
<p>The analysis uses all retained in-lab readings in nominal exposure minutes 0–179, with no time binning and no joint complete-case filter across endpoints. There were {nraw:,} source rows in that window and {nunique:,} distinct session timestamps after pooling rows sharing a timestamp. Both models use this same window. Records at elapsed minute 180 or later are excluded.</p>
<h2>Trajectory and session models</h2>
<p>Each continuous endpoint uses restricted maximum likelihood with fixed scenario, morning/afternoon, categorical visit order, scenario × time of day, elapsed time, scenario × elapsed time and time of day × elapsed time. The model uses participant, visit-within-participant and session-within-visit random intercepts. Within-session residual correlation is continuous-time AR(1), rho raised to elapsed clock minutes between observations. The fixed trajectory uses the archived elapsed exposure field. Breaks and missing observations leave real time gaps. Elapsed time is linear and slopes are per hour. Scenario slope contrasts average time of day equally and share the same afternoon slope modification.</p>
<p>Trajectory intervals and tests use a participant delete-one covariance at the full-data REML covariance parameters and a t reference with 10 degrees of freedom. This small-sample approximation allows for deviations from the fitted covariance, including different participant trajectories. Intervals are pointwise and unadjusted. Holm p values are adjusted within each outcome family and contrast type. The trajectory level contrast averages predicted levels on the minute 0–179 grid using equal time-of-day weights. With linear trajectories this is the contrast at minute 89.5.</p>
<p>The session-median column uses the visit-order-adjusted Gaussian model and small-sample plug-in t intervals, fitted to session medians calculated from exactly the same three-hour data as the trajectory model. Tonic EDA, RMSSD and LF/HF use logarithms. One negative tonic EDA reading was excluded before timestamp and session aggregation. Ratios are back-transformed, so the trajectory ratio compares modeled geometric levels. A slope ratio compares hourly multiplicative changes. Air speed, clothing and metabolic rate have no within-scenario time variation in the saved data. PMV, PPD and SET use the existing values and were not recalculated.</p>
<p><a href="AR1%20climate%20and%20physiology%20trajectories.svg">Trajectory figure</a></p>"""
    sections = [intro,"<h2>Model diagnostics</h2>",paired.table(diagnostics)]
    for (family,kind),block in results.groupby(["Family","Comparison"],sort=False):
        view = block.drop(columns=["Family","Comparison"]).dropna(axis=1,how="all")
        sections += [f"<h2>{family} · {kind}</h2>",paired.table(view)]
    sections += ["<h2>Observed session coverage near selected minutes</h2>",paired.table(coverage)]
    style = """body{font:15px/1.45 Arial;color:#182d38;margin:33px auto;max-width:1500px;padding:0 22px}
p{max-width:100ch}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:6px 8px;text-align:right;border-bottom:1px solid #dce5ea}
td:first-child,th:first-child{text-align:left}th{background:#e8f0f4}tr:nth-child(even){background:#f5f8fa}"""
    OUT.write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Trajectory analysis</title><style>{style}</style><main>{"".join(sections)}</main></html>',encoding="utf-8")


def main():
    verify_whitening()
    raw,frames = load(clean_eda=True)
    results, diagnostics, coverage, curves = fit_all(frames)
    assert len(results)==216 and results["p"].between(0,1).all()
    plot(curves)
    report(results,diagnostics,coverage,len(raw),len(frames))
    shown = results[(results["Holm p"]<.05) | (results["Outcome"].isin(["PMV","SET (°C)","Tonic EDA","Heart rate (bpm)","Wrist skin temperature (°C)","RMSSD","LF/HF"]) & results["Contrast"].isin(["T4 − T2","T4 − T1","T3 − T1"]))]
    print(shown[["Outcome","Comparison","Contrast","Trajectory estimate","95% CI low","95% CI high","Holm p","Same-window median","Same-window median Holm p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
