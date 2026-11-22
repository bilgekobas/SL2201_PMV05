"""Draw model-estimated physiology trajectories pooled and by time of day."""
from html import escape
from importlib import import_module

import numpy as np

full = import_module("18_AR1_Time_Of_Day_Interaction")
g, ar, OUT = full.g, full.ar, full.OUT
MINUTES = np.arange(180)
BANDS = (0, 30, 60, 90, 120, 150, 179)
COLORS = ("#303d47", "#17798a", "#ad582c", "#88558b")
DASHES = ("none", "7 3", "3 3", "10 3 2 3")


def predictions(beta, cov, key, periods, scenario):
    rows = np.asarray([np.mean([full.design(scenario, p, order, minute)
                               for p in periods for order in (1, 2, 3, 4)],axis=0)
                       for minute in MINUTES])
    results = np.asarray([ar.effect(beta,cov,w,key)[:3] for w in rows])
    assert np.isfinite(results).all()
    return results


def ticks_for(values):
    low, high = values.min(), values.max()
    base = 10 ** np.floor(np.log10((high-low)/4))
    step = next(base*n for n in (1,2,2.5,5,10) if base*n >= (high-low)/4)
    return np.arange(np.floor(low/step),np.ceil(high/step)+1)*step


def draw(fig,left,width,top,series,ticks):
    end=top+165

    def y(v):
        return end-(v-ticks[0])*(end-top)/(ticks[-1]-ticks[0])

    def x(t):
        return left+(t/180)*width

    for tick in ticks:
        fig.line(left,y(tick),left+width,y(tick),"#e0e8ec")
        fig.text(left-9,y(tick)+4,f"{tick:g}",12,g.MUTED,anchor="end")
    for minute in (0,30,60,90,120,150,180):
        fig.line(x(minute),top,x(minute),end,"#edf1f3")
        fig.text(x(minute),end+18,minute,12,g.MUTED,anchor="middle")
    for i,(sc,values) in enumerate(zip(ar.SCENARIOS,series)):
        color=COLORS[i]
        upper = list(zip(MINUTES,values[:,2]))
        lower = list(zip(MINUTES[::-1],values[::-1,1]))
        polygon=" ".join(f"{x(t):.2f},{y(v):.2f}" for t,v in upper+lower)
        fig.svg.append(f'<polygon points="{polygon}" fill="{color}" fill-opacity=".085"/>')
    for i,(sc,values) in enumerate(zip(ar.SCENARIOS,series)):
        line=" ".join(f"{x(t):.2f},{y(v):.2f}" for t,v in zip(MINUTES,values[:,0]))
        fig.svg.append(f'<polyline points="{line}" fill="none" stroke="{COLORS[i]}" stroke-width="2.2" stroke-dasharray="{DASHES[i]}"/>')
        for m in BANDS:
            v,lo,hi=values[m]
            text=escape(f"{sc}, minute {m}: {v:.3g} [{lo:.3g}, {hi:.3g}]")
            fig.svg.append(f'<circle cx="{x(m):.2f}" cy="{y(v):.2f}" r="5" fill="transparent"><title>{text}</title></circle>')


def render(models, panels, title, name):
    fig=g.Figure(1150,1325)
    fig.text(32,39,title,25,weight=700)
    fig.text(32,68,"Visit-order-adjusted trajectories. Shaded bands show pointwise 95% intervals.",14,g.MUTED)
    for i,sc in enumerate(ar.SCENARIOS):
        x=34+120*i
        fig.line(x,95,x+30,95,COLORS[i],2.2,DASHES[i])
        fig.text(x+38,100,sc+" (reference)" if i==0 else sc,14)
    layout=((85,1010),) if len(panels)==1 else ((80,450),(654,450))
    for (left,_),(name_,_) in zip(layout,panels):
        fig.text(left,135,name_,18,weight=700)
    for i,(key,label,unit,beta,cov) in enumerate(models):
        top=186+i*217
        fig.text(32,top-16,label+(f" ({unit})" if unit else ""),16,weight=700)
        series_by_panel=[np.asarray([predictions(beta,cov,key,periods,sc) for sc in ar.SCENARIOS])
                         for _,periods in panels]
        ticks=ticks_for(np.concatenate(series_by_panel))
        if key in ar.EXP:
            ticks=ticks[ticks>=0] if ticks[0]>=0 else np.r_[0,ticks[ticks>0]]
        for series,(left,width) in zip(series_by_panel,layout):
            draw(fig,left,width,top,series,ticks)
    fig.text(575,1249,"Elapsed exposure time (minutes)",14,anchor="middle")
    fig.text(32,1284,"All curves use minutes 0–179. Tonic EDA, RMSSD and LF/HF are back-transformed to the observed scale.",13,g.MUTED)
    fig.text(32,1307,"Intervals describe fitted levels. Direct comparisons between slopes are reported separately.",13,g.MUTED)
    g.OUT=OUT
    fig.save(name+".svg")


if __name__=="__main__":
    _,frames=ar.load(clean_eda=True)
    models=[]
    for key,title,unit,*_ in g.OUTCOMES[:5]:
        beta,cov,_=full.fit_model(frames,key)
        models.append((key,title,unit,beta,cov))
        print("Fitted",title,flush=True)
    render(models,[("Morning and afternoon weighted equally",(0,1))],
           "Pooled three-hour physiology trajectories","Final physiology - Pooled AR1 trajectories")
    render(models,[("Morning",(0,)),("Afternoon",(1,))],
           "Three-hour physiology trajectories by time of day","Final physiology - Morning and afternoon AR1 trajectories")
