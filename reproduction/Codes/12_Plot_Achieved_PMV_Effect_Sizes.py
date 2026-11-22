"""Plot the published model comparison with point estimates and 95% intervals."""
from importlib import import_module
import re

import numpy as np
import pandas as pd

graphics = import_module("02_Rebuild_Results_Figures")
OUT = graphics.OUT / "Revised manuscript"
SOURCE = OUT / "Achieved PMV model comparison.html"

PHYSIOLOGY = (
    ("Wrist skin temperature", "Difference (°C)", False, [-2, -1, 0, 1, 2]),
    ("Tonic EDA", "Tonic EDA ratio", True, [.2, .5, 1, 2, 5, 10]),
    ("Heart rate", "Difference (bpm)", False, [-15, -10, -5, 0, 5, 10]),
    ("RMSSD", "Ratio", True, [.5, .75, 1, 1.5, 2]),
    ("LF/HF", "Ratio", True, [.67, .8, 1, 1.25, 1.5, 1.75]),
)
STANDARDIZED = tuple(
    (title, "Cohen’s d", False, [-5, -2.5, 0, 2.5, 5])
    for title, *_ in PHYSIOLOGY
)
VOTES = (
    ("Thermal sensation (TSV)", "Odds ratio for a warmer vote", True, [.02, .1, 1, 10, 60]),
    ("Thermal comfort (TCV)", "Odds ratio for a more uncomfortable vote", True, [.02, .1, 1, 10, 60]),
)
MODELS = (
    ("Without PMV", graphics.BLUE, False),
    ("With PMV", graphics.ORANGE, True),
)


def parse_effect(cell):
    numbers = re.findall(r"-?\d+\.\d+", cell)
    assert len(numbers) == 3, cell
    estimate, low, high = map(float, numbers)
    assert low <= estimate <= high
    return estimate, low, high


def draw_panel(fig, table, settings, left, top):
    title, scale, ratio, ticks = settings
    fig.text(left, top, title, 19, weight=700)
    fig.text(left, top + 21, scale, 14, graphics.MUTED)
    fig.text(left + 389, top + 21, "Estimate [95% interval]", 13, graphics.MUTED)
    transform = np.log if ratio else np.asarray
    low, high = transform([ticks[0], ticks[-1]])

    def position(value):
        return left + 79 + 286 * (transform(value) - low) / (high - low)

    grid_top, grid_bottom = top + 34, top + 6 * 40 + 34
    for value in ticks:
        is_null = value == (1 if ratio else 0)
        fig.line(position(value), grid_top, position(value), grid_bottom,
                 graphics.MUTED if is_null else "#dfe7eb", 1.2 if is_null else 1,
                 "4 4" if is_null else "none")
        fig.text(position(value), grid_bottom + 18, f"{value:g}".replace("-", "−"),
                 12, graphics.MUTED, anchor="middle")

    for i, (_, row) in enumerate(table.iterrows()):
        y = grid_top + i * 40
        fig.line(left, y + 40, left + 589, y + 40, "#e8eef1")
        fig.text(left, y + 25, row["Contrast"], 14, weight=700)
        for j, (model, color, hollow) in enumerate(MODELS):
            estimate, lower, upper = parse_effect(row[model])
            assert min(ticks) <= lower <= estimate <= upper <= max(ticks), (title, row[model])
            line_y = y + 12 + j * 15
            fig.line(position(lower), line_y, position(upper), line_y, color, 1.7)
            fig.dot(position(estimate), line_y, color, hollow, 4)
            fig.text(left + 389, line_y + 4, row[model].replace("-", "−"), 13, color)


def plot(tables, settings, title, filename, footer):
    assert len(tables) == len(settings)
    rows = (len(settings) + 1) // 2
    height = 144 + 330 * rows
    fig = graphics.Figure(1280, height)
    fig.text(32, 40, title, 26, weight=700)
    fig.text(32, 66, "Each contrast compares the first scenario shown with the second.", 15, graphics.MUTED)
    for i, (model, color, hollow) in enumerate(MODELS):
        left = 34 + i * 218
        fig.line(left, 92, left + 29, 92, color, 1.7)
        fig.dot(left + 14, 92, color, hollow, 4)
        fig.text(left + 40, 97, "With continuous PMV" if i else model, 15)
    for i, (table, panel) in enumerate(zip(tables, settings)):
        assert len(table) == 6 and table.Contrast.nunique() == 6
        draw_panel(fig, table, panel, 32 + (i % 2) * 625, 133 + (i // 2) * 330)
    fig.text(32, height - 19, footer, 13, graphics.MUTED)
    graphics.OUT = OUT
    fig.save(filename)


if __name__ == "__main__":
    tables = pd.read_html(SOURCE, encoding="utf-8")
    plot(tables[1:6], PHYSIOLOGY, "Physiology with and without achieved PMV adjustment",
         "PMV adjustment - physiology effect sizes.svg",
         "88 three-hour session medians from 11 participants. Ratio axes are logarithmic. Intervals are pointwise.")
    plot(tables[6:8], VOTES, "Thermal sensation and comfort with and without achieved PMV adjustment",
         "PMV adjustment - TSV and TCV effect sizes.svg",
         "All 83 recorded endpoint questionnaires from 11 participants. Odds-ratio axes are logarithmic. Intervals are pointwise.")

    standardized = [group for _, group in tables[8].groupby("Outcome", sort=False)]
    plot(standardized, STANDARDIZED, "Standardized physiology effects with and without achieved PMV adjustment",
         "PMV adjustment - standardized physiology effect sizes.svg",
         "Cohen’s d uses the fitted residual SD of each model. Intervals are pointwise and treat the residual SD as fixed.")
