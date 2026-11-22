"""Show responses in their observed scales alongside all model contrasts."""
from html import escape
from importlib import import_module

import numpy as np
import pandas as pd

forests = import_module("12_Plot_Achieved_PMV_Effect_Sizes")
models = import_module("11_Achieved_PMV_Response_Models")
g = forests.graphics
OUT = forests.OUT
PERIODS = ("Morning", "Afternoon")
PERIOD_COLORS = ("#17798a", "#88558b")
LINE_COLORS = ("#a2c5cc", "#cab0cc")
VOTE_COLORS = {
    "tsv": ("#214c80", "#4c82bd", "#a5c6e7", "#e1e7ea", "#f3be83", "#d67b37", "#9d3f19"),
    "tcv": ("#e1e7ea", "#f3be83", "#d67b37", "#9d3f19"),
}
TICKS = ([30, 32, 34, 36, 38], [0, 5, 10, 15, 20, 25], [60, 70, 80, 90, 100],
         [50, 100, 150, 200, 250], [0, .4, .8, 1.2, 1.6])


def physiology(fig, data, key, unit, ticks, top, estimates=None):
    fig.text(32, top + 21, f"Observed session medians ({unit})" if unit else "Observed session medians", 14, g.MUTED)
    grid_top, grid_bottom = top + 78, top + 274
    people = sorted(data["part-id"].unique())
    shifts = dict(zip(people, np.linspace(-9, 9, len(people))))

    def y(value):
        return grid_bottom - (value - ticks[0]) * (grid_bottom - grid_top) / (ticks[-1] - ticks[0])

    for j, period in enumerate(PERIODS):
        left = 79 + j * 253
        positions = left + np.arange(4) * 53 + 25
        matrix = data.loc[data.cluster.eq(period)].pivot(index="part-id", columns="scenario", values=key)[g.SCENARIOS]
        fig.text(left + 103, top + 55, period, 14, PERIOD_COLORS[j], 700, "middle")
        for tick in ticks:
            fig.line(left, y(tick), left + 209, y(tick), "#dfe7eb")
            if j == 0:
                fig.text(left - 8, y(tick) + 4, f"{tick:g}", 12, g.MUTED, anchor="end")
        for person, values in matrix.iterrows():
            points = list(zip(positions + shifts[person], values))
            for (x1, a), (x2, b) in zip(points, points[1:]):
                if pd.notna(a) and pd.notna(b):
                    fig.line(x1, y(a), x2, y(b), LINE_COLORS[j])
            for x, value in points:
                if pd.notna(value):
                    assert ticks[0] <= value <= ticks[-1]
                    fig.dot(x, y(value), PERIOD_COLORS[j], r=2.7)
        for x, scenario in zip(positions, g.SCENARIOS):
            if estimates is None:
                med = y(matrix[scenario].median())
                fig.line(x - 11, med, x + 11, med, g.DARK, 2.6)
            else:
                estimate, low, high = estimates[period, scenario]
                x_effect = x + 15
                fig.line(x_effect, y(low), x_effect, y(high), g.DARK, 1.8)
                fig.dot(x_effect, y(estimate), g.DARK, hollow=True, r=4.4)
            fig.text(x, grid_bottom + 20, scenario, 13, weight=700, anchor="middle")


def votes(fig, data, key, top):
    categories = list(range(-3, 4)) if key == "tsv" else list(range(4))
    colors = VOTE_COLORS[key]
    fig.text(32, top + 21, "Observed vote distributions by time of day", 14, g.MUTED)
    for i, (category, color) in enumerate(zip(categories, colors)):
        x = 32 + 46 * i
        fig.svg.append(f'<rect x="{x}" y="{top+35}" width="13" height="13" fill="{color}"/>')
        fig.text(x+17, top+47, str(category).replace("-", "−"), 12)
    for j, period in enumerate(PERIODS):
        fig.text(32, top + 75 + j * 101, period, 14, PERIOD_COLORS[j], 700)
        for k, scenario in enumerate(g.SCENARIOS):
            observed = data.loc[data.cluster.eq(period) & data.scenario.eq(scenario), key].dropna()
            counts = observed.value_counts().reindex(categories, fill_value=0)
            assert counts.sum() == len(observed)
            y, x = top + 82 + j * 101 + k * 19, 64
            fig.text(32, y + 11, scenario, 12, weight=700)
            for category, color in zip(categories, colors):
                count = counts.loc[category]
                if count:
                    width = 432 * count / len(observed)
                    label = escape(f"{period} {scenario}, vote {category}: {count} of {len(observed)}")
                    fig.svg.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="14" fill="{color}" stroke="white" stroke-width=".6"><title>{label}</title></rect>')
                    x += width
            fig.text(505, y + 11, f"n={len(observed)}", 12, g.MUTED)
    for percent in (0, 25, 50, 75, 100):
        fig.text(64 + 4.32*percent, top + 276, f"{percent}%", 12, g.MUTED, anchor="middle")


def render(data, panels, title, filename):
    height = 170 + 328 * len(panels)
    fig = g.Figure(1250, height)
    fig.text(32, 38, title, 26, weight=700)
    fig.text(32, 72, "Absolute results for T1–T4", 17, weight=700)
    fig.text(627, 72, "Pairwise model contrasts", 17, weight=700)
    for i, (model, color, hollow) in enumerate(forests.MODELS):
        x = 629 + 193 * i
        fig.line(x, 96, x+22, 96, color, 1.7)
        fig.dot(x+11, 96, color, hollow, 4)
        fig.text(x+30, 101, "With continuous PMV" if i else model, 14)
    for i, (key, unit, ticks, table, settings) in enumerate(panels):
        top = 143 + 328 * i
        if i:
            fig.line(32, top-25, 1216, top-25, "#ccd8df")
        fig.text(32, top, settings[0], 19, weight=700)
        if key in VOTE_COLORS:
            votes(fig, data, key, top)
        else:
            physiology(fig, data, key, unit, ticks, top)
        forests.draw_panel(fig, table, ("", *settings[1:]), 627, top)
    keys = {panel[0] for panel in panels}
    if keys.difference(VOTE_COLORS):
        fig.text(32, height-39, "Lines join the same participant. Black bars show medians. Cohen’s d uses each model’s fitted residual SD.", 13, g.MUTED)
    if keys.intersection(VOTE_COLORS):
        fig.text(32, height-22, "TSV and TCV use all 83 questionnaires. Higher TCV means more discomfort. Odds-ratio axes are logarithmic.", 13, g.MUTED)
    else:
        fig.text(32, height-22, "Physiology uses 88 three-hour session medians. Intervals are pointwise and treat fitted residual SDs as fixed.", 13, g.MUTED)
    g.OUT = OUT
    fig.save(filename)


def main():
    data, _ = models.load()
    tables = pd.read_html(forests.SOURCE, encoding="utf-8")
    standardized = [block for _, block in tables[8].groupby("Outcome", sort=False)]
    physiology_panels = [(item[0], item[2], ticks, table, settings)
                         for item, ticks, table, settings in zip(g.OUTCOMES, TICKS, standardized, forests.STANDARDIZED)]
    vote_panels = [(item[0], "", None, table, settings)
                   for item, table, settings in zip(g.OUTCOMES[-2:], tables[6:8], forests.VOTES)]
    assert len(physiology_panels) == 5 and len(vote_panels) == 2
    assert all(len(panel[3]) == 6 for panel in physiology_panels + vote_panels)
    render(data, physiology_panels, "Physiology across four office scenarios", "Combined figure A - Physiology.svg")
    render(data, vote_panels, "Thermal sensation and comfort across four office scenarios", "Combined figure B - TSV and TCV.svg")
    render(data, physiology_panels + vote_panels, "Physiology, thermal sensation and comfort across four office scenarios",
           "Combined figure - Absolute responses and contrasts.svg")


if __name__ == "__main__":
    main()
