"""Sensitivity checks for scenario-dependent residual variance and visit support."""
from importlib import import_module
from pathlib import Path
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.optimize import minimize
from scipy.stats import t

m=import_module("11_Achieved_PMV_Response_Models")
OUT=Path(__file__).resolve().parents[1]/"Graphs"/"Reworked Results"/"Reviewer checks"
CRIT=t.ppf(.975,10)


def hetero(x, y, scenario, person, visit):
    """Profile REML over four residual variance ratios and two intercept ratios."""
    n,p=x.shape
    scale=y.std()
    y=(y-y.mean())/scale
    participant=np.equal.outer(person,person).astype(float)
    day=np.equal.outer(visit,visit).astype(float)
    sc=np.searchsorted(m.paired.CONDITIONS,scenario)

    def profile(parameters):
        ratios=np.expm1(parameters[:2])
        errors=np.exp(np.r_[0,parameters[2:]])
        v=np.diag(errors[sc])+ratios[0]*participant+ratios[1]*day
        chol=cho_factor(v)
        vx=cho_solve(chol,x)
        info=x.T@vx
        beta=solve(info,vx.T@y,assume_a="pos")
        residual=y-x@beta
        var=residual@cho_solve(chol,residual)/(n-p)
        objective=(n-p)*np.log(var)+2*np.log(np.diag(chol[0])).sum()+np.linalg.slogdet(info)[1]
        return objective,beta*scale,np.linalg.inv(info)*var*scale**2,np.sqrt(errors*var)*scale

    starts=([0,0,0,0,0],[1,1,0,0,0],[2,.2,.5,-.5,0])
    fits=[minimize(lambda z:profile(z)[0], start, method="L-BFGS-B",
                   bounds=[(0,np.log1p(1e4))]*2+[(-7,7)]*3,
                   options={"ftol":1e-12,"maxiter":1000}) for start in starts]
    valid=[f for f in fits if f.success]
    assert len(valid)>=2
    selected=min(valid,key=lambda f:f.fun)
    _, beta,cov,sd=profile(selected.x)
    return beta,cov,sd,len(valid),max(f.fun for f in valid)-selected.fun


def interval(b,c,w):
    value=w@b
    half=CRIT*np.sqrt(w@c@w)
    return [float(value),float(value-half),float(value+half)]


def main():
    data,_=m.load()
    rows,diag=[],[]
    for key,name in m.paired.FAMILIES["Physiology"].items():
        d=data.dropna(subset=[key])
        x=np.array([m.mixed.row(sc,p=="Afternoon",o,True) for sc,p,o in zip(d.scenario,d.cluster,d.order)])
        y=m.response_transform(key,d[key].to_numpy())
        person=d["part-id"].to_numpy()
        visit=d.visit.to_numpy()
        sc=d.scenario.to_numpy()
        beta,cov,_=m.mixed.gaussian(x,y,person,visit)
        hb,hc,sd,starts,gap=hetero(x,y,sc,person,visit)
        k=d.order.to_numpy()!=1
        contrast_keep=np.array([i for i in range(x.shape[1]) if i!=5])
        restricted=x[k][:,contrast_keep]
        assert len(d.loc[k])==66 and np.linalg.matrix_rank(restricted)==restricted.shape[1]
        rb,rc,_=m.mixed.gaussian(restricted,y[k],person[k],visit[k])
        diag.append([name,*sd.tolist(),starts,gap])
        for kind,label,w in m.mixed.contrasts(True):
            if kind=="Scenario":
                triples=[interval(beta,cov,w),interval(hb,hc,w),interval(rb,rc,w[contrast_keep])]
                if key in m.paired.LOG:
                    triples=np.exp(triples).tolist()
                rows.append([name,label,*triples])
        print(name,rows[-6:],flush=True)
    (OUT/"session sensitivity checks.json").write_text(json.dumps({"rows":rows,"diagnostics":diag},indent=2),encoding="utf8")


if __name__=="__main__":
    main()
