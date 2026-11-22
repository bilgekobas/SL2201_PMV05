"""Draw pooled submission figures for exposure and physiological responses."""
from importlib import import_module
import numpy as np
import pandas as pd

annotated=import_module("15_Annotated_Physiology")
main=annotated.main_figure
models,g=main.models,main.g
OUT=main.forests.OUT.parent/"Reviewer checks"


def exposure(data):
    fig=g.Figure(1100,435)
    fig.text(32,40,"Achieved PMV and SET across the four configurations",24,weight=700)
    fig.text(32,67,"Each point is a three-hour session median. Short bars show the scenario median.",14,g.MUTED)
    for i,(key,title,ticks) in enumerate((("pmv","A  PMV",np.arange(.2,.71,.1)),
                                          ("SET_C","B  SET (°C)",np.arange(25,27.51,.5)))):
        left=77+i*530
        fig.text(left,111,title,19,weight=700)
        top,bottom=136,366
        def y(v):return bottom-(v-ticks[0])*(bottom-top)/(ticks[-1]-ticks[0])
        for v in ticks:
            fig.line(left,y(v),left+438,y(v),"#dce5ea")
            fig.text(left-9,y(v)+4,f"{v:g}",13,g.MUTED,anchor="end")
        if i==0:
            fig.line(left,y(.5),left+438,y(.5),g.MUTED,1.2,"5 4")
            fig.text(left+5,y(.5)-7,"Design target +0.5",12,g.MUTED)
        for j,sc in enumerate(g.SCENARIOS):
            x=left+54+110*j
            values=data.loc[data.scenario.eq(sc),key].sort_values()
            assert len(values)==22
            offsets=np.random.default_rng(j+545).permutation(np.linspace(-19,19,len(values)))
            for dx,v in zip(offsets,values):
                fig.dot(x+dx,y(v),g.BLUE,r=3.2)
            yc=y(values.median())
            fig.line(x-20,yc,x+20,yc,g.DARK,2.7)
            fig.text(x,bottom+22,sc,14,weight=700,anchor="middle")
    fig.text(32,416,"22 sessions per scenario. PMV was the design metric and SET the secondary exposure index.",13,g.MUTED)
    g.OUT=OUT
    fig.save("submission pooled exposure.svg")


def physiology(data):
    fig=g.Figure(920,2030)
    fig.text(30,38,"Pooled physiological responses across the four configurations",24,weight=700)
    fig.dot(35,69,g.BLUE);fig.text(48,74,"Individual participant",14)
    fig.dot(250,69,g.DARK,hollow=True);fig.text(263,74,"Mixed-model estimate and 95% interval",14)
    fig.text(30,99,"Brackets compare the higher-numbered scenario with the lower-numbered scenario.",13,g.MUTED)
    tables=pd.read_html(main.forests.SOURCE,encoding="utf-8")
    centers=annotated.CENTERS
    for i,((key,title,unit,*_),ticks,table,scale) in enumerate(zip(g.OUTCOMES[:5],main.combined.TICKS,tables[1:6],annotated.SCALES)):
        top=135+371*i
        if i:fig.line(30,top-22,880,top-22,"#d5e0e5")
        fig.text(30,top,f"{chr(65+i)}  {title}"+(f" ({unit})" if unit else ""),18,weight=700)
        a,b=top+20,top+177
        def y(v):return b-(v-ticks[0])*(b-a)/(ticks[-1]-ticks[0])
        for tick in ticks:
            fig.line(79,y(tick),856,y(tick),"#e0e8ec")
            fig.text(68,y(tick)+4,f"{tick:g}",12,g.MUTED,anchor="end")
        raw=data.assign(value=models.response_transform(key,data[key].to_numpy()))
        observed=raw.pivot_table(index="part-id",columns="scenario",values="value").reindex(columns=g.SCENARIOS)
        assert observed.shape==(11,4)
        if key in models.paired.LOG:observed=np.exp(observed)
        for dx,(_,values) in zip(np.linspace(-12,12,len(observed)),observed.iterrows()):
            pts=list(zip(np.array(list(centers.values()))-9+dx,values))
            for (x1,v1),(x2,v2) in zip(pts,pts[1:]):fig.line(x1,y(v1),x2,y(v2),"#aac8d3")
            for x,v in pts:fig.dot(x,y(v),g.BLUE,r=2.8)
        estimates=main.group_estimates(data,key,pooled=True)
        for sc,x in centers.items():
            est,lo,hi=estimates["Pooled",sc]
            assert ticks[0]<=lo<=est<=hi<=ticks[-1]
            fig.line(x+17,y(lo),x+17,y(hi),g.DARK,2)
            fig.dot(x+17,y(est),g.DARK,hollow=True,r=4.4)
            fig.text(x,b+19,sc,14,weight=700,anchor="middle")
        fig.text(80,top+218,f"{scale} [95% interval]",13,g.MUTED)
        contrasts=table.set_index("Contrast")["Without PMV"]
        for j,(first,second) in enumerate(annotated.PAIRS):
            x1,x2=centers[second],centers[first]
            h=top+241+j*27
            xmid=(x1+x2)/2
            for x in (x1,x2):fig.line(x,h-5,x,h,g.MUTED)
            fig.line(x1,h,xmid-81,h,g.MUTED)
            fig.line(xmid+81,h,x2,h,g.MUTED)
            fig.text(xmid,h+4,contrasts[f"{first} − {second}"].replace("-","−"),13,weight=700,anchor="middle")
    fig.text(30,1983,"Dots combine the two session medians per participant and configuration. Lines connect the same participant.",13,g.MUTED)
    fig.text(30,2007,"The three log-modeled outcomes are combined on the log scale and displayed in their original units.",13,g.MUTED)
    g.OUT=OUT
    fig.save("submission pooled physiology.svg")


def timeline():
    fig=g.Figure(1100,180)
    fig.text(30,34,"Schedule of one scenario visit",25,weight=700)
    labels=("Seated rest and sensor setup","Exposure 1","Lunch break","Exposure 2")
    detail=("30 minutes","3 hours","1.5 hours","3 hours")
    spans=(230,310,190,310)
    colors=("#dfe9ed","#cee4ed","#efebe2","#cee4ed")
    x=30
    for label,d,width,color in zip(labels,detail,spans,colors):
        fig.svg.append(f'<rect x="{x}" y="61" width="{width-4}" height="59" fill="{color}"/>')
        fig.text(x+13,86,label,15,weight=700)
        fig.text(x+13,106,d,13,g.MUTED)
        x+=width
    fig.text(30,150,"Arrival at approximately 09:00. Both exposures used the same office configuration.",13,g.MUTED)
    g.OUT=OUT
    fig.save("submission visit schedule.svg")


if __name__=="__main__":
    data,_=models.load()
    exposure(data)
    physiology(data)
    timeline()
