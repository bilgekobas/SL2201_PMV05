"""Annotate paired physiology plots with selected primary model contrasts."""
from importlib import import_module

import numpy as np
import pandas as pd

main_figure = import_module("14_Main_Physiology_Figure")
combined = main_figure.combined
forests, g = combined.forests, combined.g
PAIRS = (("T2", "T1"), ("T4", "T2"), ("T3", "T1"), ("T4", "T1"))
CENTERS = dict(zip(g.SCENARIOS, (167, 370, 573, 776)))
SCALES = ("Difference in °C", "Tonic EDA ratio", "Difference in bpm", "RMSSD ratio", "LF/HF ratio")


def draw_outcome(fig, data, key, title, unit, ticks, top, table, scale):
    fig.text(30, top, title + (f" ({unit})" if unit else ""), 18, weight=700)
    top_grid, bottom_grid = top + 20, top + 177

    def y(value):
        return bottom_grid - (value - ticks[0]) / (ticks[-1] - ticks[0]) * (bottom_grid - top_grid)

    for tick in ticks:
        fig.line(79, y(tick), 856, y(tick), "#e0e8ec")
        fig.text(68, y(tick)+4, f"{tick:g}", 12, g.MUTED, anchor="end")
    estimates = main_figure.group_estimates(data, key, pooled=True)
    for j, period in enumerate(combined.PERIODS):
        matrix = data.loc[data.cluster.eq(period)].pivot(index="part-id", columns="scenario", values=key)[g.SCENARIOS]
        xs = np.array(list(CENTERS.values())) + (-25, 25)[j]
        for dx, (_, row) in zip(np.linspace(-9, 9, len(matrix)), matrix.iterrows()):
            points = list(zip(xs + dx, row))
            for (x1, a), (x2, b) in zip(points, points[1:]):
                if pd.notna(a) and pd.notna(b):
                    fig.line(x1, y(a), x2, y(b), combined.LINE_COLORS[j])
            for x, value in points:
                if pd.notna(value):
                    fig.dot(x, y(value), combined.PERIOD_COLORS[j], r=2.6)
    for scenario, x in CENTERS.items():
        estimate, low, high = estimates["Pooled", scenario]
        assert ticks[0] <= low <= estimate <= high <= ticks[-1]
        fig.line(x, y(low), x, y(high), g.DARK, 2)
        fig.dot(x, y(estimate), g.DARK, hollow=True, r=4.8)
        fig.text(x, bottom_grid+19, scenario, 14, weight=700, anchor="middle")

    fig.text(80, top+218, f"{scale} [95% interval]", 13, g.MUTED)
    contrasts = table.set_index("Contrast")["Without PMV"]
    for j, (first, second) in enumerate(PAIRS):
        x1, x2 = CENTERS[second], CENTERS[first]
        y_line = top+241+j*27
        label = contrasts[f"{first} − {second}"].replace("-", "−")
        midpoint = (x1+x2)/2
        gap = 81
        fig.line(x1, y_line-5, x1, y_line, g.MUTED)
        fig.line(x2, y_line-5, x2, y_line, g.MUTED)
        fig.line(x1, y_line, midpoint-gap, y_line, g.MUTED)
        fig.line(midpoint+gap, y_line, x2, y_line, g.MUTED)
        fig.text(midpoint, y_line+4, label, 13, weight=700, anchor="middle")


def main():
    data, _ = main_figure.models.load()
    tables = pd.read_html(forests.SOURCE, encoding="utf-8")
    fig = g.Figure(920, 2030)
    fig.text(30, 38, "Physiology across the four office scenarios", 25, weight=700)
    for j, period in enumerate(combined.PERIODS):
        x = 35 + 147*j
        fig.dot(x, 69, combined.PERIOD_COLORS[j])
        fig.text(x+12, 74, period, 14)
    fig.dot(333, 69, g.DARK, hollow=True)
    fig.text(345, 74, "Pooled fitted level and 95% interval", 14)
    fig.text(30, 99, "Brackets show the higher-numbered scenario relative to the lower-numbered scenario.", 13, g.MUTED)
    for i, ((key, title, unit, *_), ticks, table, scale) in enumerate(zip(
            g.OUTCOMES[:5], combined.TICKS, tables[1:6], SCALES)):
        top = 135 + 371*i
        if i:
            fig.line(30, top-22, 880, top-22, "#d5e0e5")
        draw_outcome(fig, data, key, title, unit, ticks, top, table, scale)
    fig.text(30, 1983, "Colored points are three-hour session medians. Lines join the same participant within each time of day.", 13, g.MUTED)
    fig.text(30, 2007, "Fitted levels and contrasts use the primary model without PMV adjustment and weight morning and afternoon equally.", 13, g.MUTED)
    g.OUT = forests.OUT
    fig.save("Main physiology - Pooled estimates and pairwise comparisons.svg")


if __name__ == "__main__":
    main()
