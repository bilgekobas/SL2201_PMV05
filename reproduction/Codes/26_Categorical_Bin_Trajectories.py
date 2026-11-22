from importlib import import_module
from itertools import combinations, product
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

ar=import_module("05_AR1_Trajectory_Analysis")
g=ar.figures
signals=g.OUTCOMES[:5]
periods=("Morning","Afternoon")
starts=range(0,180,30)
cells=list(product(ar.SCENARIOS,periods,starts))
pairs=list(combinations(ar.SCENARIOS,2))
selected=[("T1","T2"),("T1","T3"),("T1","T4"),("T2","T4")]
out=ar.paired.OUT.parent/"Categorical 30-minute trajectories"
out.mkdir(exist_ok=True)
g.OUT=out
colors=dict(zip(ar.SCENARIOS,("#22181C","#168282","#a56800","#d00038")))

def design(sc,period,start,order=1):
    return np.r_[1,[item==(sc,period,start) for item in cells[1:]],[order==j for j in (2,3,4)]]

def paired_response(key):
    rows=frames[ar.KEY+["order","visit","session","time-elapsed",key]].copy()
    rows["start"]=(rows["time-elapsed"]//30).astype(int)*30
    groups=rows.groupby(ar.KEY+["order","visit","session","start"],as_index=False)[key]
    d=groups.agg(["median","count"])
    d=d.loc[d["count"].ge(10)].rename(columns={"median":key})
    d["unix"]=d.start*60
    d=d.sort_values(ar.KEY+["start"]).reset_index(drop=True)
    x=np.asarray([design(s,p,b,o) for s,p,b,o in zip(d.scenario,d.cluster,d.start,d.order)])
    assert np.linalg.matrix_rank(x)==x.shape[1]
    beta,cov,diag=ar.trajectory(d,x,ar.transform(key,d[key].to_numpy()))
    diag["30-minute AR correlation"]=diag["One-minute AR correlation"]**30
    return beta,cov,diag,d

def contrasts(key,beta,cov):
    rows=[]
    for start in starts:
        for sc in ar.SCENARIOS:
            m,a=(design(sc,p,start)-design(sc,p,0) for p in periods)
            add(rows,key,beta,cov,"Within-session change",sc,start,{"Morning":m,"Afternoon":a,"Pooled":(a+m)/2,"Afternoon minus morning":a-m})
        for first,last in pairs:
            m,a=((design(last,p,start)-design(last,p,0))-(design(first,p,start)-design(first,p,0)) for p in periods)
            add(rows,key,beta,cov,"Difference in changes",f"{last} − {first}",start,{"Morning":m,"Afternoon":a,"Pooled":(a+m)/2,"Afternoon minus morning":a-m})
    return rows

def add(rows,key,beta,cov,comparison,contrast,start,weights):
    for p,w in weights.items():
        if start:
            est,lo,hi,_=ar.effect(beta,cov,w,key)
        else:
            est=lo=hi=float(key in ar.EXP)
        rows.append([key,comparison,contrast,start,p,est,lo,hi,"Ratio" if key in ar.EXP else "Difference"])

def figure(key,name,unit,results,n):
    fig=g.Figure(1330,860)
    fig.text(32,40,name,25,weight=700)
    fig.text(32,70,"Scenario differences in change from minutes 0–29. Points and whiskers show estimates and pointwise 95% CIs.",14)
    fig.text(32,92,"The right column compares the afternoon scenario difference with the morning scenario difference.",14)
    labels=["Morning","Afternoon","Afternoon vs morning"]
    for j,p in enumerate(labels):fig.text(297+405*j,127,p,17,weight=700,anchor="middle")
    grouped=results.loc[results.Signal.eq(key)&results.Comparison.eq("Difference in changes")]
    transform=np.log if key in ar.EXP else np.asarray
    inverse=np.exp if key in ar.EXP else float
    chosen=grouped.loc[grouped.Contrast.isin(f"{b} − {a}" for a,b in selected)&grouped.Period.ne("Pooled")]
    bounds=transform(chosen[["CI low","CI high"]].to_numpy())
    lo,hi=min(bounds.min(),0),max(bounds.max(),0)
    pad=(hi-lo)*.05
    lo,hi=lo-pad,hi+pad
    base=10**np.floor(np.log10((hi-lo)/4))
    step=min((base*x for x in (1,2,2.5,5,10)),key=lambda s:abs(s-(hi-lo)/4))
    ticks=np.arange(np.ceil(lo/step),np.floor(hi/step)+1)*step
    for i,(a,b) in enumerate(selected):
        top=159+i*163
        fig.text(36,top+11,f"{b} − {a}",14,colors[b],700)
        for j,p in enumerate(("Morning","Afternoon","Afternoon minus morning")):
            left=135+405*j
            def x(m):return left+65*m/30
            def y(v):return top+110-(v-lo)*110/(hi-lo)
            for v in ticks:
                fig.line(left,y(v),left+325,y(v),"#dedede")
                fig.text(left-8,y(v)+4,f"{inverse(v):.3g}",11,anchor="end")
            fig.line(left,y(0),left+325,y(0),"#888888",1.1,"4 4")
            rows=chosen.loc[chosen.Period.eq(p)&chosen.Contrast.eq(f"{b} − {a}")].sort_values("Bin start")
            previous=None
            for _,r in rows.iterrows():
                start=int(r["Bin start"])
                center=x(start)
                est,lower,upper=transform(r[["Estimate","CI low","CI high"]].to_numpy(dtype=float))
                if previous is not None:fig.line(*previous,center,y(est),colors[b],2)
                previous=(center,y(est))
                if start:
                    fig.line(center,y(lower),center,y(upper),colors[b],1.6)
                    fig.dot(center,y(est),colors[b],r=3.6)
                else:fig.dot(center,y(0),"#777777",hollow=True,r=3)
                fig.text(center,top+130,str(start),11,anchor="middle")
    scale="Ratios on a log axis. Right column divides the afternoon ratio by the morning ratio." if key in ar.EXP else f"Effects in {unit}. Right column subtracts the morning effect from the afternoon effect."
    fig.text(32,822,"Horizontal axis shows the starting minute of each 30-minute bin. Open point marks the reference bin.",13)
    fig.text(32,844,scale+f" Fit uses {n} eligible session bins.",13)
    file={"skin-temp-wrist-smooth":"Wrist temperature","eda_tonic":"Tonic EDA","hr-clean":"Heart rate","rmssd":"RMSSD","lf/hf":"LF-HF"}[key]
    fig.save(file+".svg")

if __name__=="__main__":
    _,frames=ar.load(clean_eda=True)
    ar.verify_whitening()
    results=[]
    diagnostics=[]
    coverage=[]
    for key,name,unit,*_ in signals:
        beta,cov,diag,d=paired_response(key)
        assert d.session.nunique()==88 and d["part-id"].nunique()==11
        results.extend(contrasts(key,beta,cov))
        diagnostics.append({"Signal":key,"Eligible bins":len(d),**diag})
        coverage.append(d.groupby(["scenario","cluster","start"]).size().rename("Eligible sessions").reset_index().assign(Signal=key))
        print(key,len(d),diag,flush=True)
    r=pd.DataFrame(results,columns=["Signal","Comparison","Contrast","Bin start","Period","Estimate","CI low","CI high","Scale"])
    assert len(r)==1200 and np.isfinite(r[["Estimate","CI low","CI high"]]).all().all()
    r.to_csv(out/"Effect sizes and intervals.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"Model diagnostics.csv",index=False)
    pd.concat(coverage).to_csv(out/"Bin coverage.csv",index=False)
    for (key,name,unit,*_),row in zip(signals,diagnostics):figure(key,name,unit,r,row["Eligible bins"])
    (out/"Analysis notes.md").write_text("""# Categorical 30-minute trajectory models

Six 30-minute bins cover minutes 0–179. Each participant-session bin is summarized by its median when it contains at least ten unique timestamps. Nonpositive tonic EDA was removed. Tonic EDA, RMSSD and LF/HF are modeled on the log scale. The REML model includes the full scenario × period × bin interaction and categorical visit order. Random intercepts account for participant, visit within participant, and exposure session. Residual AR(1) correlation accounts for spacing between bins within each session.

Each effect compares the indicated bin with the first bin. Scenario pairs compare those changes. The afternoon-versus-morning interaction compares the scenario difference in change across periods. Pooled contrasts equally weight the two periods. Log-scale effects are exponentiated. Intervals use participant-deletion covariance with covariance parameters held fixed and a t reference with 10 degrees of freedom. Intervals are pointwise. The exploratory analyses include many correlated contrasts. The first bin is the reference by construction. The models identify changes during the observed exposure periods.

The five SVGs show four selected scenario pairs. The CSV includes all six pairs, within-scenario changes, period-specific effects, pooled effects and interactions for all six bins.
""",encoding="utf8")
    with ZipFile(out.with_suffix(".zip"),"w",ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.suffix in (".csv", ".svg", ".md"):
                z.write(p,p.name)
        assert z.testzip() is None
