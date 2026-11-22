"""Participant-paired effect sizes and binned exposure trajectories."""
from importlib import import_module
from itertools import combinations
from math import exp, lgamma, log, sqrt

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nct, t
from scipy.special import ndtr, roots_genlaguerre
from functools import lru_cache

package = import_module("06_Revised_Manuscript_Figures")
trajectory, figures, mixed, paired = package.trajectory, package.figures, package.mixed, package.paired
OUT = package.OUT / "Effect-size figures and tables"
SCENARIOS = paired.CONDITIONS
PERIODS = paired.PERIODS
KEY = paired.KEY
EXP = trajectory.EXP
VOTES = paired.ORDINAL
REFERENCE_MINUTES = 30
COLORS = trajectory.COLORS
RNG = np.random.default_rng(22817)


@lru_cache(maxsize=11)
def t_quadrature(df):
    nodes, weights = roots_genlaguerre(128, df/2-1)
    return np.sqrt(2*nodes/df), weights/weights.sum()


def noncentrality(test, df, prob):
    # The normal/chi-square mixture avoids noncentral-t CDF overflow in
    # tightly controlled climate contrasts with very large standardized effects.
    scale, weights = t_quadrature(df)
    centers = test*scale
    bound = abs(test)*max(scale.max(),1)+12
    return brentq(lambda v:weights @ ndtr(centers-v)-prob, -bound, bound)


def interval(values, key, interaction=False):
    """Return an interpretable estimate and bias-corrected paired magnitude."""
    x = np.asarray(values.dropna(), dtype=float)
    n = len(x)
    if not n:
        return {"n":0}
    mean = x.mean()
    tx = np.exp if key in EXP else float
    row = {"n":n,"Estimate":tx(mean),"95% low":np.nan,"95% high":np.nan,
           "Paired g":np.nan,"g 95% low":np.nan,"g 95% high":np.nan}
    if key in VOTES:
        if n >= 3 and np.ptp(x) > 0:
            sample = x[RNG.integers(0,n,size=(16000,n))].mean(axis=1)
            row["95% low"],row["95% high"] = np.quantile(sample,[.025,.975])
        return row
    if n < 3 or np.ptp(x) < 1e-10:
        return row
    sd = x.std(ddof=1)
    df = n-1
    se = sd/sqrt(n)
    half = t.ppf(.975,df)*se
    row["95% low"],row["95% high"] = tx(mean-half),tx(mean+half)
    correction = exp(lgamma(df/2)-.5*log(df/2)-lgamma((df-1)/2))
    test = mean/se
    row["Paired g"] = correction*mean/sd
    for field, prob in (("g 95% low",.975),("g 95% high",.025)):
        nc = noncentrality(test,df,prob)
        row[field] = nc/sqrt(n)
    return row


def scale(key, interaction=False):
    if key == "eda_tonic":
        return "Tonic EDA ratio"
    if key in EXP:
        return "Ratio"
    if key in VOTES:
        return "Net higher-vote probability"
    return "Difference"


def make_row(family,key,label,kind,contrast,values,**meta):
    return {"Family":family,"Outcome":label,"Comparison":kind,"Contrast":contrast,
            "Scale":scale(key),**meta,**interval(values,key)}


def session_effects(session):
    output = []
    for family, outcomes in paired.FAMILIES.items():
        for key,label in outcomes.items():
            w = trajectory.transform(key,session[key]).unstack(["scenario","cluster"])
            for a,b in combinations(SCENARIOS,2):
                delta = pd.DataFrame({p:w[b,p]-w[a,p] for p in PERIODS})
                signed = np.sign(delta) if key in VOTES else delta
                output.append(make_row(family,key,label,"Scenario",f"{b} − {a}",signed.mean(axis=1)))
                if key not in VOTES:
                    output.append(make_row(family,key,label,"Scenario by time",f"{b} − {a}",
                                           delta["Afternoon"]-delta["Morning"]))
            times = {}
            for sc in SCENARIOS:
                raw = w[sc,"Afternoon"]-w[sc,"Morning"]
                times[sc] = np.sign(raw) if key in VOTES else raw
                output.append(make_row(family,key,label,"Time within scenario",sc,times[sc]))
            output.append(make_row(family,key,label,"Time across scenarios","Afternoon − Morning",
                                   pd.DataFrame(times).mean(axis=1)))
    return pd.DataFrame(output)


def bin_effects(frames, width):
    keys = [k for outcomes in trajectory.FAMILIES.values() for k in outcomes]
    d = frames[KEY+["time-elapsed",*keys]].copy()
    d["bin"] = (d["time-elapsed"]//width).astype(int)*width
    first = d.loc[d["time-elapsed"].lt(REFERENCE_MINUTES)].groupby(KEY)[keys]
    baseline = first.median().where(first.count().ge(10))
    group = d.loc[d["bin"].ge(30)].groupby(KEY+["bin"])[keys]
    follow = group.median().where(group.count().ge(10 if width==30 else 4))
    records = []
    for family,outcomes in trajectory.FAMILIES.items():
        for key,label in outcomes.items():
            for start in range(30,180,width):
                after = follow[key].xs(start,level="bin")
                delta = trajectory.transform(key,after)-trajectory.transform(key,baseline[key])
                wide = delta.unstack(["scenario","cluster"])
                window = f"{start}–{start+width-1}"
                for period in PERIODS:
                    for sc in SCENARIOS:
                        records.append(make_row(family,key,label,"Change from first 30 minutes",sc,
                                               wide[sc,period],Time=period,Bin=window,Start=start))
                for a,b in combinations(SCENARIOS,2):
                    by_period = pd.DataFrame({p:wide[b,p]-wide[a,p] for p in PERIODS})
                    for period in PERIODS:
                        records.append(make_row(family,key,label,"Difference in change",f"{b} − {a}",
                                               by_period[period],Time=period,Bin=window,Start=start))
                    records.append(make_row(family,key,label,"Difference in change",f"{b} − {a}",
                                           by_period.mean(axis=1),Time="Available times averaged",Bin=window,Start=start))
    return pd.DataFrame(records)


def effect_plot(session_results):
    fig = figures.Figure(1050,550)
    fig.text(32,40,"Paired scenario effects relative to T1",25,weight=700)
    fig.text(32,66,"Dots are participant-paired effects. Bars are 95% intervals.",color=figures.MUTED)
    for j,(sc,col) in enumerate(zip(SCENARIOS[1:],COLORS[1:])):
        x=34+70*j
        fig.dot(x,90,col,r=4)
        fig.text(x+12,95,sc,13)
    panels = [
        (210,420,list(paired.FAMILIES["Physiology"].values()),"A   Physiology","Bias-corrected paired standardized change", "Paired g","g 95% low","g 95% high"),
        (779,230,list(paired.FAMILIES["Questionnaire"].values()),"B   Questionnaire","Net probability of a higher vote","Estimate","95% low","95% high"),
    ]
    for left,width,outcomes,title,axis,est,lo,hi in panels:
        sub = session_results.loc[session_results.Outcome.isin(outcomes)&session_results.Comparison.eq("Scenario")&
                                  session_results.Contrast.str.endswith("T1")]
        if est=="Paired g":
            extent=max(1.5,np.ceil(max(-sub[lo].min(),sub[hi].max())*2)/2)
        else:
            extent=1
        fig.text(left,128,title,17,weight=700)
        xaxis=figures.x_axis(fig,left,width,148,438,-extent,extent,np.linspace(-extent,extent,5),0)
        for i,label in enumerate(outcomes):
            y=177+55*i
            fig.text(left-11,y+4,label.replace(" (vote points)",""),12,anchor="end")
            cell=sub.loc[sub.Outcome.eq(label)].set_index("Contrast")
            for j,(sc,col) in enumerate(zip(SCENARIOS[1:],COLORS[1:])):
                row=cell.loc[f"{sc} − T1"]
                ypos=y+13*(j-1)
                if pd.notna(row[lo]):
                    fig.line(xaxis(row[lo]),ypos,xaxis(row[hi]),ypos,col,1.8)
                if pd.notna(row[est]):
                    fig.dot(xaxis(row[est]),ypos,col,r=4)
        fig.text(left+width/2,475,axis,12,anchor="middle")
    fig.text(32,522,"Higher votes indicate warmer sensation or more discomfort. Models and all six scenario contrasts are tabulated separately.",12,figures.MUTED)
    fig.save("Paired scenario effect sizes.svg")


def bin_plot(results, width, filename):
    fig = figures.Figure(1080,730)
    fig.text(34,42,f"Change from the first 30 minutes · {width}-minute bins",25,weight=700)
    fig.text(34,70,"Lines show mean paired changes by scenario. Small gray bars show pointwise 95% intervals.",color=figures.MUTED)
    for i,(sc,color) in enumerate(zip(SCENARIOS,COLORS)):
        x=36+i*85
        fig.line(x,100,x+22,100,color,3)
        fig.text(x+28,105,sc,13)
    for col,period in enumerate(PERIODS):
        fig.text(286+col*505,144,period,18,weight=700,anchor="middle")
    labels = [("Wrist skin temperature (°C)","Wrist temperature change (°C)",0),
              ("Tonic EDA","Tonic EDA ratio",1)]
    for row,(label,title,null) in enumerate(labels):
        top=180+250*row
        sub=results.loc[results.Outcome.eq(label)&results.Comparison.eq("Change from first 30 minutes")]
        minimum=min(sub["95% low"].min(),null)
        maximum=max(sub["95% high"].max(),null)
        step=.5
        low=np.floor(minimum/step)*step
        high=np.ceil(maximum/step)*step
        fig.text(34,top,title,16,weight=700)
        for col,period in enumerate(PERIODS):
            left=87+505*col
            def coords(start,value,index):
                return left+((start+width/2-30)/150)*380+3*(index-1.5),top+184-(value-low)/(high-low)*155
            for value in np.arange(low,high+step/2,step):
                y=coords(30,value,1.5)[1]
                fig.line(left,y,left+380,y,"#dfe7eb" if value!=null else "#84939b",1)
                fig.text(left-7,y+4,f"{value:g}",11,anchor="end")
            for j,(sc,color) in enumerate(zip(SCENARIOS,COLORS)):
                a=sub.loc[sub.Time.eq(period)&sub.Contrast.eq(sc)].sort_values("Start")
                previous=None
                for _,v in a.iterrows():
                    if pd.isna(v.Estimate):
                        previous=None
                        continue
                    x,y=coords(v.Start,v.Estimate,j)
                    if pd.notna(v["95% low"]):
                        fig.line(x,coords(v.Start,v["95% low"],j)[1],x,coords(v.Start,v["95% high"],j)[1],"#a6afb5")
                    if previous is not None:
                        fig.line(*previous,x,y,color,1.6)
                    previous=(x,y)
                    fig.dot(x,y,color,r=3.4)
            for start in range(30,181,30):
                x=left+(start-30)/150*380
                fig.text(x,top+204,start,11,anchor="middle")
    fig.text(540,653,"Elapsed minutes",13,anchor="middle")
    fig.text(34,688,"First 30 minutes serve as the reference for both bin widths. Tables give participant counts and direct scenario contrasts.",12,figures.MUTED)
    fig.text(34,708,"No data after minute 179 enter these plots.",12,figures.MUTED)
    fig.save(filename)


def report(session_results, model, bins, sensitivity):
    style = """body{font:15px/1.5 Arial;color:#182d38;max-width:1500px;margin:34px auto;padding:0 24px}
p,li{max-width:98ch}h2{margin-top:32px}table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:5px 7px;border-bottom:1px solid #dce5ea;text-align:right}th{background:#e8f0f4}td:first-child,th:first-child{text-align:left}tr:nth-child(even){background:#f5f8fa}"""
    model = model.drop(columns=["Model p","Model Holm p"])
    join=["Family","Outcome","Comparison","Contrast"]
    comparisons=session_results.merge(model,on=join,how="outer",validate="one_to_one")
    sections=["""<h1>Effect sizes for the three-hour PMV study</h1>
<p>Continuous outcomes use participant-session medians over elapsed minutes 0–179 with breaks excluded. All 83 endpoint questionnaires are used. Scenarios for two conflicting questionnaire labels follow the matching visit log. Model-adjusted estimates retain their original units, ratios or ordinal odds ratios. The paired columns summarize observed within-participant differences. They do not adjust for the imbalance in visit order, so use the model columns when interpreting an order-adjusted effect.</p>
<p>For a scenario comparison, differences are formed at the same time of day and averaged equally across each participant’s available morning/afternoon matches. Continuous standardized effects divide the mean by the SD of the participant differences and apply the small-sample Hedges correction. This is a paired effect size (g derived from d_z). The original manuscript standardized selected model effects by residual SD, so those values are on a different scale. Continuous intervals use a t reference in native or transformed units and noncentral-t inversion for standardized effects, assuming approximately normal participant differences. Tonic EDA, RMSSD and LF/HF use log. One negative tonic EDA reading was excluded before aggregation. Ratios are back-transformed.</p>
<p>For votes, the paired effect is P(higher) minus P(lower), with ties contributing zero and participant bootstrap intervals. This preserves the ordered categories. The adjusted ordinal odds ratio is reported alongside it where available. No interval is shown for fewer than three participants or a zero observed spread. Intervals are pointwise and unadjusted for multiplicity. Outcomes with fixed scenario inputs are summarized directly in the manuscript exposure table. No standardized effect or p-value is required for a fixed input.</p>
<p><a href="Paired%20scenario%20effect%20sizes.svg">Paired scenario effect-size figure</a> · <a href="30-minute%20baseline%20changes.svg">30-minute trajectory figure</a> · <a href="10-minute%20baseline%20changes.svg">10-minute sensitivity figure</a></p>
<h2>Binned trajectories</h2>
<p>Each point uses a per-session median for its plotted bin, compared with that same session’s median for minutes 0–29. At least 10 valid timestamps are required in the reference and in each 30-minute bin, and at least four in a 10-minute bin. The 10-minute plots retain the same reference window to isolate the choice of follow-up bin width. Participant counts can change across bins when sample coverage is insufficient. Bin centres are used on the time axis.</p>
<p>The bin tables show both change within a scenario and direct effects comparing those changes across scenarios at the same time of day. The table also shows all six direct scenario comparisons with available time-of-day contrasts averaged within participant. Within each bin, the unit for uncertainty is the participant. The bin estimates remain correlated across time and are intended as a description of the shape, not as independent tests. The previous unbinned AR(1) model provides a separate check on the linear trajectory and accounts for serial correlation.</p>"""]
    for kind, group in comparisons.groupby("Comparison",sort=False):
        view=group.drop(columns=["Comparison"]).dropna(axis=1,how="all")
        sections += [f"<h2>Session effects · {kind}</h2>",paired.table(view)]
    for width, data in [(30,bins),(10,sensitivity)]:
        for (family, kind),group in data.groupby(["Family","Comparison"],sort=False):
            sections += [f"<details><summary>{width}-minute bins · {family} · {kind}</summary>",
                         paired.table(group.drop(columns=["Family","Comparison","Start"])), "</details>"]
    (OUT/"Effect-size results.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Effect-size results</title>'
        f"<style>{style}details{{margin:12px 0}}summary{{font-weight:700;cursor:pointer}}</style><main>{''.join(sections)}</main>",
        encoding="utf-8")


def main():
    for df in (4,10):
        for test in (-3.,0.,3.):
            for target in (.025,.975):
                solved = noncentrality(test,df,target)
                assert abs(nct.cdf(test,df,solved)-target) < 2e-4
    OUT.mkdir(exist_ok=True)
    figures.OUT=OUT
    raw,frames=trajectory.load(clean_eda=True)
    measures=frames.drop(columns=["visit","order","unix","time-elapsed","session"])
    sensor=measures.groupby(KEY).median()
    votes,_=package.session_votes()
    session=sensor.join(votes)
    assert len(session)==88 and session.tsv.count()==83
    effects=session_effects(session)
    bins=bin_effects(frames,30)
    sensitivity=bin_effects(frames,10)
    assert len(bins)==18*5*(8+6*3) and len(sensitivity)==18*15*(8+6*3)
    effect_plot(effects)
    bin_plot(bins,30,"30-minute baseline changes.svg")
    bin_plot(sensitivity,10,"10-minute baseline changes.svg")
    visits=raw[["part-id","scenario","visit","order"]].drop_duplicates()
    model,_=mixed.analyze(session,visits)
    model.loc[model.Outcome.eq("Tonic EDA"),"Model scale"]="Tonic EDA ratio"
    report(effects,model,bins,sensitivity)
    print(effects.loc[effects.Comparison.eq("Scenario")&effects.Contrast.isin(["T4 − T2","T3 − T1"])&
          effects.Outcome.isin(["SET (°C)","PMV","Tonic EDA","RMSSD","Thermal sensation vote","Thermal comfort vote"])]
          .round(3).to_string(index=False))
    for b in (bins,sensitivity):
        print(b.loc[b.Outcome.isin(["Wrist skin temperature (°C)","Tonic EDA"])&b.Contrast.eq("T4 − T2")&
             b.Time.eq("Available times averaged"),["Outcome","Bin","n","Estimate","95% low","95% high","Paired g"]]
             .round(3).to_string(index=False))


if __name__=="__main__":
    main()
