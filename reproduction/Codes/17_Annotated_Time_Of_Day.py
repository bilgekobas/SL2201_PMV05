"""Annotate both types of simple effect and export all interaction contrasts."""
from importlib import import_module

import numpy as np
import pandas as pd

time = import_module("16_Time_Of_Day_Physiology_Figure")
g, combined, models = time.g, time.combined, time.models
OUT = combined.OUT
CENTERS = dict(zip(g.SCENARIOS, (155, 390, 625, 860)))
# Draw shorter brackets nearest the fitted levels.
DISPLAY_PAIRS = (("T1", "T2"), ("T2", "T4"), ("T1", "T3"), ("T1", "T4"))
BRACKET_ORDER = tuple(time.PAIRS.index(pair) for pair in DISPLAY_PAIRS)


def label(interval):
    return models.formatted(interval).replace("-", "−")


def draw(fig, top, key, title, unit, ticks, scale, level, within, cross):
    fig.text(30, top, title + (f" ({unit})" if unit else ""), 19, weight=700)
    fig.text(30, top+24, f"Scenario comparisons above · {scale} [95% interval]", 13, g.MUTED)
    chart_top, chart_bottom = top+173, top+326
    ratio = key in models.paired.LOG
    back = np.exp if ratio else np.asarray

    def y(value):
        assert ticks[0] <= value <= ticks[-1]
        return chart_bottom-(value-ticks[0])/(ticks[-1]-ticks[0])*(chart_bottom-chart_top)

    for tick in ticks:
        fig.line(84, y(tick), 930, y(tick), "#e0e8ec")
        fig.text(74, y(tick)+4, f"{tick:g}", 12, g.MUTED, anchor="end")
    for rank, i in enumerate(BRACKET_ORDER):
        a, b = time.PAIRS[i]
        for j, color in enumerate(time.COLORS):
            shift = (-10, 10)[j]
            x1, x2 = CENTERS[a]+shift, CENTERS[b]+shift
            center = (x1+x2)/2
            h = chart_top-17-30*rank-14*j
            text = label(np.exp(cross[i,j]) if ratio else cross[i,j])
            fig.line(x1, h, center-84, h, color, 1.1)
            fig.line(center+84, h, x2, h, color, 1.1)
            for x in (x1, x2):
                fig.line(x, h, x, h+4, color, 1.1)
            fig.text(center, h+4, text, 12, color, 700, "middle")
    for i, (scenario, x) in enumerate(CENTERS.items()):
        for j, (period, color) in enumerate(zip(time.PERIODS, time.COLORS)):
            dot = x+(-10,10)[j]
            values = back(level[scenario, period])
            est, lo, hi = values
            time.tooltip(fig, f"{scenario} {period}", values)
            fig.line(dot, y(lo), dot, y(hi), color, 2)
            fig.dot(dot, y(est), color, hollow=bool(j), r=4.5)
            fig.svg.append("</g>")
        fig.text(x, chart_bottom+23, scenario, 14, weight=700, anchor="middle")
        fig.line(x-22, chart_bottom+35, x+22, chart_bottom+35, g.MUTED)
        for end in (x-22, x+22):
            fig.line(end, chart_bottom+31, end, chart_bottom+35, g.MUTED)
        fig.text(x, chart_bottom+52, label(np.exp(within[i]) if ratio else within[i]), 13, g.DARK, 700, "middle")
    fig.text(30, chart_bottom+73, f"Within each scenario below · Afternoon relative to morning, {scale.lower()} [95% interval]",
             13, g.MUTED)


def main():
    data, _ = models.load()
    fig = g.Figure(1010, 2290)
    fig.text(30, 36, "Physiological responses by scenario and time of day", 25, weight=700)
    for j, (period, color) in enumerate(zip(time.PERIODS, time.COLORS)):
        x = 36+145*j
        fig.dot(x, 65, color, hollow=bool(j))
        fig.text(x+13, 70, period, 14)
    fig.text(330, 70, "Fitted levels and pointwise 95% intervals", 14, g.MUTED)
    fig.text(30, 97, "Upper brackets compare the higher-numbered scenario with the lower-numbered scenario within one period.", 13, g.MUTED)
    interaction_rows = []
    for i, ((key, title, unit, *_), ticks, scale) in enumerate(zip(g.OUTCOMES, combined.TICKS, time.LABELS)):
        top = 131 + i*420
        level, within, cross, change = time.estimate(data, key)
        draw(fig, top, key, title, unit, ticks, scale, level, within, cross)
        ratio = key in models.paired.LOG
        for (a, b), period_effects, effect in zip(time.PAIRS, cross, change):
            if ratio:
                period_effects, effect = np.exp(period_effects), np.exp(effect)
            interaction_rows.append({"Outcome": title, "Scenario pair": f"{b} − {a}",
                                     "Morning [95% CI]": label(period_effects[0]),
                                     "Afternoon [95% CI]": label(period_effects[1]),
                                     "Interaction [95% CI]": label(effect),
                                     "Interaction scale": "Ratio of ratios" if ratio else f"Difference of differences ({unit})"})
    fig.text(30, 2244, "The joint model uses 88 three-hour session medians, accounts for visit order and does not adjust for achieved PMV.", 13, g.MUTED)
    fig.text(30, 2267, "Brackets show original-unit differences or ratios. Tonic EDA is fitted on the log scale.", 13, g.MUTED)
    g.OUT = OUT
    fig.save("Main physiology - Annotated time-of-day comparisons.svg")
    table = models.paired.table(pd.DataFrame(interaction_rows))
    (OUT/"Physiology time-of-day interaction table.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Physiology time-of-day interactions</title>'
        '<style>body{font:15px/1.5 Arial;color:#182d38;max-width:1200px;margin:32px auto;padding:0 20px}'
        'table{width:100%;border-collapse:collapse}th,td{padding:7px;text-align:left;border-bottom:1px solid #dce5ea}'
        'th{background:#e8f0f4}p{max-width:87ch}</style>'
        '<h1>Scenario-by-time-of-day interactions</h1>'
        '<p>Each row reports the same scenario contrast in the morning and afternoon and its change between periods. '
        'For temperature and heart rate the interaction is the afternoon contrast minus the morning contrast. '
        'For log-modeled outcomes it is the afternoon scenario ratio divided by the morning scenario ratio. '
        'Intervals are pointwise 95% intervals from the joint model and account for covariance between contrasts. '
        'The model adjusts for visit order and uses the primary specification without achieved PMV.</p>'+table, encoding="utf-8")


if __name__ == "__main__":
    main()
