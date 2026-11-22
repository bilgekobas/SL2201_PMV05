"""Fit the full three-way AR(1) trajectory interaction for physiology."""
from importlib import import_module
from itertools import combinations
import html

import numpy as np
import pandas as pd

time = import_module("16_Time_Of_Day_Physiology_Figure")
ar = time.models.trajectory
g = time.g
OUT = time.combined.OUT
PAIRS = tuple(combinations(ar.SCENARIOS, 2))


def design(scenario, period, order, minute):
    elapsed = (minute - 89.5) / 60
    return np.r_[ar.design(scenario, period, order, minute),
                 [period * elapsed * (scenario == sc) for sc in ar.SCENARIOS[1:]]]


def weights(sc, period, slope=False):
    if slope:
        return design(sc, period, 1, 120) - design(sc, period, 1, 60)
    return np.mean([design(sc, period, order, 89.5) for order in (1, 2, 3, 4)], axis=0)


def fit_model(frames, key):
    d = frames.dropna(subset=[key]).sort_values(ar.KEY+["unix"]).reset_index(drop=True)
    x = np.array([design(sc, int(period == "Afternoon"), order, minute)
                  for sc, period, order, minute in zip(d.scenario, d.cluster, d.order, d["time-elapsed"])])
    assert np.linalg.matrix_rank(x) == x.shape[1]
    beta, cov, diagnostics = ar.trajectory(d, x, time.models.response_transform(key, d[key].to_numpy()))
    diag = {"Rows":len(d), "Sessions":d.session.nunique(), **diagnostics}
    return beta, cov, diag


def fit(frames, key):
    beta, cov, diag = fit_model(frames, key)

    def effect(w):
        return np.asarray(ar.effect(beta,cov,w,key)[:3])

    level_rows, pair_rows, time_rows = [], [], []
    for a, b in PAIRS:
        low, high = [weights(b, p) - weights(a, p) for p in (0, 1)]
        early, late = [weights(b, p, True) - weights(a, p, True) for p in (0, 1)]
        pair = f"{b} − {a}"
        level_rows.append((pair, effect((low+high)/2), effect(low), effect(high), effect(high-low)))
        pair_rows.append((pair, effect((early+late)/2), effect(early), effect(late), effect(late-early)))
    for sc in ar.SCENARIOS:
        early, late = [weights(sc,p,True) for p in (0,1)]
        time_rows.append((sc, effect(weights(sc,1)-weights(sc,0)), effect(early), effect(late), effect(late-early)))
    return level_rows, pair_rows, time_rows, diag


def table(rows, columns):
    return ar.paired.table(pd.DataFrame([(label, *[f"{a:.3f} [{b:.3f}, {c:.3f}]" for a,b,c in vals])
                                          for label,*vals in rows],columns=columns))


def main():
    ar.verify_whitening()
    _, frames = ar.load(clean_eda=True)
    med = pd.read_html(time.models.OUT, encoding="utf-8")
    diagnostics, sections, figure_values = [], [], []
    for i,(key,name) in enumerate(ar.paired.FAMILIES["Physiology"].items()):
        print("Fitting",name,flush=True)
        levels, slopes, periods, diag = fit(frames,key)
        diagnostics.append({"Outcome":name,**diag})
        figure_values.append((name,key,slopes))
        scale = "Tonic EDA ratios" if key=="eda_tonic" else "Ratios" if key in ar.EXP else "Original-unit differences"
        ses = med[i+1][["Contrast","Without PMV","With PMV"]]
        sections.extend([f"<h2>{html.escape(name)}</h2><p>{scale}. Brackets give pointwise 95% intervals.</p>",
                         "<h3>1. Pooled session-median model</h3>", ar.paired.table(ses),
                         "<h3>2. AR(1) trajectory levels and average slopes</h3>",
                         table([(l[0],l[1],s[1]) for l,s in zip(levels,slopes)],
                               ["Pair","Three-hour average level","Pooled slope contrast per hour"]),
                         "<h3>3. Morning and afternoon levels</h3>",
                         table([(r[0],r[1]) for r in periods],["Scenario","Afternoon vs morning at the midpoint"]),
                         table([(l[0],*l[2:]) for l in levels],["Pair","Morning level","Afternoon level","Change in level contrast"]),
                         "<h3>4. Morning and afternoon trajectories</h3>",
                         table([(r[0],*r[2:]) for r in periods],["Scenario","Morning slope","Afternoon slope","Change in slope"]),
                         table([(s[0],*s[2:]) for s in slopes],["Pair","Morning slope contrast","Afternoon slope contrast","Change in slope contrast"])])
        for pair, _, morning, afternoon, change in slopes:
            print(name,pair,"| slope AM, PM; interaction",np.round(morning,3),np.round(afternoon,3),np.round(change,3))
        for pair, _, _, _, change in levels:
            print("LEVEL",name,pair,np.round(change,3))
    fig = g.Figure(1030, 1110)
    fig.text(28,39,"Do scenario slopes change between morning and afternoon?",24,weight=700)
    fig.text(28,66,"Afternoon contrast relative to morning contrast, per elapsed hour. Bars show pointwise 95% intervals.",13,g.MUTED)
    for i,(name,key,slopes) in enumerate(figure_values):
        top=108+i*191
        left, width = 160, 575
        ratio = key in ar.EXP
        intervals = np.array([row[-1] for row in slopes])
        plot = np.log(intervals) if ratio else intervals
        extent=np.ceil(np.abs(plot).max()/(.1 if ratio else .5))*(.1 if ratio else .5)

        def x(v):
            return left+width*(v+extent)/(2*extent)

        scale = "ratio of slope ratios" if ratio else "°C per hour" if key.startswith("skin") else "bpm per hour"
        fig.text(28,top,f"{name} · {scale}",16,weight=700)
        for tick in np.linspace(-extent,extent,5):
            fig.line(x(tick),top+13,x(tick),top+147,g.MUTED if tick==0 else "#dfe7eb",1.1,"4 4" if tick==0 else "none")
            fig.text(x(tick),top+165,f"{np.exp(tick) if ratio else tick:.2g}".replace("-","−"),12,g.MUTED,anchor="middle")
        for j,((pair,*_), (est,lo,hi),(a,b,c)) in enumerate(zip(slopes,plot,intervals)):
            y=top+25+22*j
            fig.text(left-12,y+4,pair,13,weight=700,anchor="end")
            fig.line(x(lo),y,x(hi),y,g.BLUE,1.8)
            fig.dot(x(est),y,g.BLUE,r=3.8)
            fig.text(763,y+4,f"{a:.3f} [{b:.3f}, {c:.3f}]".replace("-","−"),13)
    fig.text(28,1081,"Ratio axes are logarithmic. Dashed lines indicate no change.",13,g.MUTED)
    g.OUT=OUT
    fig.save("AR1 physiology - Time-of-day slope interactions.svg")
    (OUT/"AR1 physiology - Time-of-day trajectory analysis.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Three-hour physiology trajectory interactions</title>'
        '<style>body{font:15px/1.5 Arial;color:#182d38;max-width:1200px;margin:32px auto;padding:0 24px}'
        'table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;text-align:left;border-bottom:1px solid #dce5ea}'
        'th{background:#e8f0f4}p{max-width:100ch}</style><h1>Three-hour physiology trajectory interactions</h1>'
        '<p>Gaussian REML with categorical scenario, time of day, categorical visit order, elapsed hours and all interactions '
        'between scenario, time of day and elapsed hours. Participant, visit and session random intercepts and continuous-time AR(1) '
        'residuals within session account for repeated observations. Minutes 0–179 only; breaks excluded. The model uses the primary '
        'specification without achieved PMV. One nonpositive tonic EDA reading was excluded. Intervals use the participant delete-one covariance with covariance parameters held fixed '
        'and a t reference with 10 degrees of freedom.</p>'
        '<p>Levels are evaluated at the center of the three-hour window. Equal morning and afternoon weights define the pooled '
        'model effects. For logged signals a slope is a multiplicative change per hour. A pairwise slope contrast divides those '
        'multipliers, and the interaction divides the afternoon pairwise slope ratio by its morning counterpart. Nulls are zero '
        'for differences or one for ratios. These are exploratory, pointwise intervals.</p>'
        '<p><a href="AR1%20physiology%20-%20Time-of-day%20slope%20interactions.svg">Slope interaction figure</a></p>'
        +''.join(sections)+'<h2>Fit diagnostics</h2>'+ar.paired.table(pd.DataFrame(diagnostics)),encoding="utf-8")


if __name__ == "__main__":
    main()
