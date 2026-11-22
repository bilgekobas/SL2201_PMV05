"""Build exposure and model checks for the resubmission supplement."""
from importlib import import_module
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t

m=import_module("11_Achieved_PMV_Response_Models")
a=m.trajectory
full=import_module("18_AR1_Time_Of_Day_Interaction")
g=a.figures
OUT=Path(__file__).resolve().parents[1]/"Graphs"/"Reworked Results"/"Reviewer checks"
SC=list(a.SCENARIOS)
PAIRS=list(combinations(SC,2))
PERIODS=("Morning","Afternoon")
CRIT=t.ppf(.975,10)


def session_fit(d,key):
    x=np.array([m.mixed.row(sc,p=="Afternoon",o,True) for sc,p,o in zip(d.scenario,d.cluster,d.order)])
    y=m.response_transform(key,d[key].to_numpy())
    beta,cov,sd=m.mixed.gaussian(x,y,d["part-id"].to_numpy(),d.visit.to_numpy())
    zp=pd.get_dummies(d["part-id"],dtype=float).to_numpy()
    zv=pd.get_dummies(d.visit,dtype=float).to_numpy()
    zpz=np.column_stack([zp*sd[0],zv*sd[1]])
    root=np.linalg.solve(np.eye(zpz.shape[1])*sd[2]**2+zpz.T@zpz,zpz.T@(y-x@beta))
    fitted=x@beta+zpz@root
    return beta,cov,sd,fitted,y-fitted,root[:zp.shape[1]],root[zp.shape[1]:]


def exposure(d):
    fig=g.Figure(1150,830)
    fig.text(30,40,"Achieved exposure and matched scenario differences",24,weight=700)
    fig.text(30,67,"Three-hour session medians and same-participant comparisons within morning or afternoon.",14,g.MUTED)
    fig.dot(34,94,g.BLUE);fig.text(46,99,"Morning",13)
    fig.dot(145,94,g.ORANGE,hollow=True);fig.text(157,99,"Afternoon",13)
    fig.dot(275,94,g.DARK);fig.text(287,99,"Visit-order-adjusted difference and 95% interval",13)
    output=[]
    for i,(key,title,ticks,dticks) in enumerate((
        ("pmv","PMV",np.arange(.2,.71,.1),np.arange(-.4,.41,.2)),
        ("SET_C","SET (°C)",np.arange(25,27.51,.5),np.arange(-1.6,1.61,.8)),
    )):
        top=143+330*i
        fig.text(32,top,title+" session medians",18,weight=700)
        fig.text(570,top,title+" matched contrasts",18,weight=700)
        grid_top,grid_bottom=top+27,top+267
        def y(v): return grid_bottom-(v-ticks[0])*(grid_bottom-grid_top)/(ticks[-1]-ticks[0])
        for v in ticks:
            fig.line(84,y(v),501,y(v),"#dce5ea")
            fig.text(76,y(v)+4,f"{v:g}",13,g.MUTED,anchor="end")
        if i==0:
            fig.line(84,y(.5),501,y(.5),g.DARK,1.3,"5 4")
            fig.text(86,y(.5)-7,"Design target +0.5",12,g.MUTED)
        for j,sc in enumerate(SC):
            x=130+j*105
            for k,(period,color) in enumerate(zip(PERIODS,(g.BLUE,g.ORANGE))):
                v=d.loc[d.scenario.eq(sc)&d.cluster.eq(period),key].sort_values()
                for dx,val in zip(np.linspace(-10,10,len(v)),v):
                    fig.dot(x+22*(k-.5)+dx,y(val),color,hollow=bool(k),r=3)
            fig.text(x,grid_bottom+21,sc,14,weight=700,anchor="middle")
        axis=g.x_axis(fig,659,340,grid_top,grid_bottom,dticks[0],dticks[-1],np.round(dticks,3)+0.,0)
        beta,cov,_,*_=session_fit(d,key)
        matched=d.pivot(index=["part-id","cluster"],columns="scenario",values=key)
        for j,((aa,bb),(_,label,w)) in enumerate(zip(PAIRS,(z for z in m.mixed.contrasts(True) if z[0]=="Scenario"))):
            assert label==f"{bb} − {aa}"
            center=grid_top+20+j*40
            fig.text(650,center+4,label,13,weight=700,anchor="end")
            v=matched[bb]-matched[aa]
            for k,(period,color) in enumerate(zip(PERIODS,(g.BLUE,g.ORANGE))):
                points=v.xs(period,level="cluster").sort_values()
                for dy,value in zip(np.linspace(-4,4,len(points)),points):
                    assert dticks[0]<=value<=dticks[-1]
                    fig.dot(axis(value),center-11+10*k+dy,color,hollow=bool(k),r=2.8)
            est=w@beta
            width=CRIT*np.sqrt(w@cov@w)
            assert dticks[0]<=est-width and est+width<=dticks[-1]
            fig.line(axis(est-width),center+11,axis(est+width),center+11,g.DARK,1.7)
            fig.dot(axis(est),center+11,g.DARK,r=3)
            value=f"{est:.2f} [{est-width:.2f}, {est+width:.2f}]"
            fig.text(1014,center+15,value,12)
            output.append([title,label,float(v.min()),float(v.max()),value])
    fig.text(30,796,"The PMV design target applies to the left panel only. Zero denotes no difference between scenarios.",13,g.MUTED)
    fig.text(30,818,"Each scenario pair contributes 11 morning and 11 afternoon differences.",13,g.MUTED)
    g.OUT=OUT
    fig.save("Matched PMV and SET with 95 percent intervals.svg")
    return output


def diagnostic_axis(fig,x,top,xx,yy,kind):
    left,right=x+38,x+261
    y0,y1=top+24,top+145
    if kind=="QQ":
        v=max(np.abs(xx).max(),np.abs(yy).max(),.5)*1.06
        lo,hi=-v,v
        l2,h2=lo,hi
    else:
        lo,hi=xx.min(),xx.max()
        margin=max((hi-lo)*.06,.01)
        lo-=margin; hi+=margin
        v=max(np.abs(yy).max(),.5)*1.06
        l2,h2=-v,v
    px=lambda u:left+(u-lo)*(right-left)/(hi-lo)
    py=lambda u:y1-(u-l2)*(y1-y0)/(h2-l2)
    fig.line(left,y1,right,y1,"#cbd9e0")
    fig.line(left,y0,left,y1,"#cbd9e0")
    if kind=="QQ":
        fig.line(px(lo),py(lo),px(hi),py(hi),"#86959c",1,"4 3")
    else:
        fig.line(left,py(0),right,py(0),"#86959c",1,"4 3")
    for a_,b_ in zip(xx,yy): fig.dot(px(a_),py(b_),g.BLUE,r=2.3)
    for val in (lo,(lo+hi)/2,hi):
        fig.text(px(val),y1+15,f"{val:.2g}",10,g.MUTED,anchor="middle")
    for val in (l2,0,h2):
        fig.text(left-4,py(val)+4,f"{val:.2g}",10,g.MUTED,anchor="end")


def qq(v):
    z=np.sort(v)
    sd=z.std(ddof=1)
    return norm.ppf((np.arange(len(z))+.5)/len(z)),(z-z.mean())/sd if sd else z


def ar_check(frames,key):
    beta,_,diag=full.fit_model(frames,key)
    rho=diag["One-minute AR correlation"]
    sd=np.array([diag[x] for x in ("Participant SD","Visit SD","Session SD")])
    keep=[]
    d=frames.dropna(subset=[key]).sort_values(a.KEY+["unix"])
    for _,d in d.groupby("part-id",sort=False):
        x=np.array([full.design(sc,p=="Afternoon",v,minute) for sc,p,v,minute in zip(d.scenario,d.cluster,d.order,d["time-elapsed"])])
        y=m.response_transform(key,d[key].to_numpy())-x@beta
        z=np.column_stack([np.full(len(d),sd[0]),pd.get_dummies(d.visit,dtype=float).to_numpy()*sd[1],
                           pd.get_dummies(d.session,dtype=float).to_numpy()*sd[2]])
        gap=d.groupby("session").unix.diff().to_numpy()/60
        ix=np.flatnonzero(np.isfinite(gap))
        phi=rho**gap[ix]
        divisor=np.sqrt(1-phi**2)
        q,j=y.copy(),z.copy()
        q[ix]=(y[ix]-phi*y[ix-1])/divisor
        j[ix]=(z[ix]-phi[:,None]*z[ix-1])/divisor[:,None]
        eps=diag["Residual SD"]
        bhat=np.linalg.solve(j.T@j+np.eye(j.shape[1])*eps**2,j.T@q)
        e=(q-j@bhat)/eps
        contiguous=np.flatnonzero((gap==1)&np.r_[False,gap[:-1]==1])
        keep.append(np.column_stack([e[contiguous-1],e[contiguous]]))
    z=np.concatenate(keep)
    return [diag["Rows"],rho,float(np.corrcoef(z.T)[0,1]),diag["Optimized starts"],diag["REML objective gap across starts"]]


def main():
    OUT.mkdir(exist_ok=True)
    d,votes=m.load()
    e=exposure(d)
    fig=g.Figure(1170,1080)
    fig.text(30,37,"Diagnostics for Gaussian physiology session models",23,weight=700)
    for i,name in enumerate(("Conditional residual Q–Q","Residuals versus fitted","Participant intercept Q–Q","Visit intercept Q–Q")):
        fig.text(32+285*i,76,name,15,weight=700)
    ss,ar=[],[]
    _,frames=a.load(clean_eda=True)
    for i,(key,title) in enumerate(a.paired.FAMILIES["Physiology"].items()):
        beta,cov,sd,fitted,resid,person,visit=session_fit(d,key)
        top=104+i*183
        fig.text(32,top,title,15,weight=700)
        checks=[qq(resid),(fitted,resid/sd[2]),qq(person),qq(visit)]
        for j,((xx,yy),kind) in enumerate(zip(checks,("QQ","resid","QQ","QQ"))):
            diagnostic_axis(fig,32+285*j,top,xx,yy,kind)
        group=pd.Series(resid,index=d.scenario).groupby(level=0).std()
        ss.append([title,*group.reindex(SC).to_list(),float(group.max()/group.min())])
        ar.append([title,*ar_check(frames,key)])
        print(title,ss[-1],ar[-1],flush=True)
    fig.text(30,1049,"Intercept plots use standardized empirical predictions and are descriptive, particularly for 11 participants.",13,g.MUTED)
    fig.text(30,1068,"Conditional residuals retain the model scale. EDA, RMSSD and LF/HF were modeled on the log scale.",13,g.MUTED)
    g.OUT=OUT
    fig.save("Session model diagnostics.svg")
    model=pd.read_html(m.OUT,encoding="utf-8")[-1]
    model=model.loc[model.Outcome.isin(["Thermal sensation vote","Thermal comfort vote"])]
    categories={key:{str(k):int(v) for k,v in d[key].value_counts().sort_index().items()} for key in ("tsv","tcv")}
    (OUT/"reviewer diagnostic values.json").write_text(json.dumps({"exposure":e,"session":ss,"ar":ar,"ordinal":model.to_dict("records"),"votes":categories},indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
