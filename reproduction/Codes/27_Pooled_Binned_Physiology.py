"""Reproduce the descriptive pooled 30-minute physiology curves in Figure 5."""

from importlib import import_module
import base64
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

bins = import_module("07_Effect_Sizes_And_Binned_Trajectories")
ar, g = bins.trajectory, bins.figures
INK, GRAY, GRID = "#121417", "#555555", "#e4e4e4"
SC = ["#22181C", "#239998", "#ad6900", "#EA002D"]
outcomes = g.OUTCOMES[:5]


def ticks(values, target=5, ratio=False):
    lo,hi=np.nanmin(values),np.nanmax(values)
    spread=hi-lo
    d=spread/target
    floor=10**np.floor(np.log10(d))
    gap=min((floor*x for x in (1,2,2.5,4,5,10)),key=lambda s:abs(s-d))
    lower=lo-.04*spread
    if ratio:lower=max(0,lower)
    upper=hi+.04*spread
    interior=np.arange(np.ceil(lower/gap),np.floor(upper/gap)+1)*gap
    return lower,upper,interior


def yaxis(fig,left,right,top,bottom,values,null=None,ratio=False):
    lo,hi,labels=ticks(values,ratio=ratio)
    def y(v):return bottom-(v-lo)*(bottom-top)/(hi-lo)
    for v in labels:
        fig.line(left,y(v),right,y(v),GRID)
        fig.text(left-8,y(v)+4,f'{v:.5g}',12,GRAY,anchor='end')
    if null is not None:fig.line(left,y(null),right,y(null),'#898989',1.1,'4 4')
    return y


def add_header(fig,title,subtitle):
    fig.text(34,42,title,25,INK,700)
    fig.text(34,69,subtitle,14,GRAY)


def binned(frames, width):
    keys=[x[0] for x in outcomes]
    d=frames[bins.KEY+['time-elapsed',*keys]].copy()
    d['Start']=(d['time-elapsed']//width).astype(int)*width
    first=d.loc[d['time-elapsed'].lt(30)].groupby(bins.KEY)[keys]
    baseline=first.median().where(first.count().ge(10))
    follow=d.loc[d.Start.ge(30)].groupby(bins.KEY+['Start'])[keys]
    follow_values=follow.median().where(follow.count().ge(10 if width==30 else 4))
    result=[]
    for key in keys:
        for start in range(30,180,width):
            delta=ar.transform(key,follow_values[key].xs(start,level='Start'))-ar.transform(key,baseline[key])
            for sc in g.SCENARIOS:
                observations=delta.xs(sc,level='scenario').unstack('cluster')
                for period,values in [*[(p,observations[p]) for p in bins.PERIODS],
                                      ('Pooled',observations[list(bins.PERIODS)].mean(axis=1,skipna=False))]:
                    result.append(dict(Outcome=key,Start=start,Period=period,Scenario=sc,**bins.interval(values,key)))
    return pd.DataFrame(result)


def binned_plot(result,width,periods,filename):
    split=len(periods)==2
    height=1165 if split else 813
    fig=g.Figure(1240,height)
    heading='Within-session physiology by time of day' if split else 'Physiological changes over three hours'
    add_header(fig,heading,f'{width}-minute bins relative to the first 30 minutes. Points show participant means with pointwise 95% CIs.')
    for i,(sc,col) in enumerate(zip(g.SCENARIOS,SC)):
        x=38+100*i
        fig.line(x,96,x+26,96,col,3)
        fig.dot(x+13,96,col,r=3.2)
        fig.text(x+33,101,sc,13,INK,700)
    if split:
        for j,period in enumerate(periods):fig.text(318+j*608,132,period,17,INK,700,'middle')
    for i,(key,signal,unit,*_) in enumerate(outcomes):
        is_ratio=key in ar.EXP
        null=1 if is_ratio else 0
        sub=result.loc[result.Outcome.eq(key)&result.Period.isin(periods)]
        vals=[null,*sub[['Estimate','95% low','95% high']].to_numpy().ravel()]
        for j,period in enumerate(periods):
            left=70+608*(j if split else i%2)
            top=164+196*i if split else 150+209*(i//2)
            y=yaxis(fig,left,left+504,top,top+123,vals,null,ratio=is_ratio)
            fig.text(left-34,top-12,signal+(' ratio' if is_ratio and not signal.endswith('ratio') else
                     f' change ({unit})' if not is_ratio else ' relative to baseline'),16,INK,700)
            def x(v):return left+(v-30)/150*504
            for m in range(30,181,30):
                fig.text(x(m),top+144,m,11,GRAY,anchor='middle')
            for k,(sc,color) in enumerate(zip(g.SCENARIOS,SC)):
                path=[]
                for row in sub.loc[sub.Period.eq(period)&sub.Scenario.eq(sc)].sort_values('Start').itertuples(index=False,name=None):
                    v=dict(zip(sub.columns,row))
                    if not np.isfinite(v['Estimate']):
                        path=[]
                        continue
                    xx=x(v['Start']+width/2)+(k-1.5)*2.1
                    yy=y(v['Estimate'])
                    if path:fig.line(*path[-1],xx,yy,color,2.3)
                    path.append((xx,yy))
                    lo,hi=v['95% low'],v['95% high']
                    if np.isfinite(lo) and np.isfinite(hi):
                        fig.line(xx,y(lo),xx,y(hi),'#a0a0a0',.8)
                    fig.dot(xx,yy,color,r=3.4)
    fig.text(34,height-18,'Elapsed exposure minutes. '+('Each period has its own baseline.' if split else 'Both periods are weighted equally within each participant. Ratios use the log scale before back-transformation.'),13,GRAY)
    fig.save(filename)

def main():
    _, frames = ar.load(clean_eda=True)
    out = ar.paired.OUT.parent / "Categorical 30-minute trajectories"
    out.mkdir(parents=True, exist_ok=True)
    g.OUT = out
    filename = "Figure 05 - Pooled 30-minute physiology changes.svg"
    estimates = binned(frames, 30)
    binned_plot(estimates, 30, ["Pooled"], filename)
    estimates.loc[estimates.Period.eq("Pooled")].to_csv(out / "Figure 05 descriptive values.csv", index=False)
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    root = ET.parse(out / filename).getroot()
    group = root.find(f"{{{ns}}}g")
    group.set("font-family", "JBE Archivo, Archivo SemiCondensed, Arial, sans-serif")
    css = []
    for weight, name in ((400, "Regular"), (700, "Bold")):
        font_directory = next(
            directory for directory in (g.ROOT / "Fonts", g.ROOT.parent / "Fonts")
            if (directory / "Archivo_SemiCondensed-Regular.ttf").exists()
        )
        font = base64.b64encode((font_directory / f"Archivo_SemiCondensed-{name}.ttf").read_bytes()).decode()
        css.append(f'@font-face{{font-family:"JBE Archivo";font-weight:{weight};'
                   f'src:url(data:font/ttf;base64,{font}) format("truetype");}}')
    style = ET.Element(f"{{{ns}}}style")
    style.text = "".join(css)
    root.insert(0, style)
    ET.ElementTree(root).write(out / filename, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
