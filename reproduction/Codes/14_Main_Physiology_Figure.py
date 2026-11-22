"""Draft a main-text physiology figure with individual data and model summaries."""
from importlib import import_module

import numpy as np
import pandas as pd
from scipy.stats import t

combined = import_module("13_Combined_Response_Figure")
models, forests, g = combined.models, combined.forests, combined.g


def group_estimates(data, key, pooled=False):
    """Prediction on the fitted scale, equally weighted across the four visit orders."""
    d = data.dropna(subset=[key])
    design = np.array([models.mixed.row(sc, day == "Afternoon", order, True)
                       for sc, day, order in zip(d.scenario, d.cluster, d.order)])
    beta, cov, _ = models.mixed.gaussian(design, models.response_transform(key, d[key].to_numpy()),
                                         d["part-id"].to_numpy(), d.visit.to_numpy())
    critical = t.ppf(.975, d["part-id"].nunique()-1)
    if key in models.paired.LOG:
        back_transform = np.exp
    else:
        back_transform = np.asarray
    periods = {"Pooled": combined.PERIODS} if pooled else {period: (period,) for period in combined.PERIODS}
    return {
        (label, scenario): models.effect((beta, cov), np.mean([
            models.mixed.row(scenario, period == "Afternoon", order, True)
            for period in included for order in (1, 2, 3, 4)
        ], axis=0), back_transform, critical)
        for label, included in periods.items() for scenario in g.SCENARIOS
    }


def contrast_panel(fig, table, top, ticks, ratio, scale):
    left, right = 675, 973
    axis_top, axis_bottom = top + 68, top + 236
    transform = np.log if ratio else np.asarray
    lo, hi = transform([ticks[0], ticks[-1]])
    fig.text(600, top + 21, scale, 14, g.MUTED)
    for tick in ticks:
        x = left + (right-left)*(transform(tick)-lo)/(hi-lo)
        null = tick == (1 if ratio else 0)
        fig.line(x, axis_top, x, axis_bottom, g.MUTED if null else "#e2e9ec", 1.2 if null else 1, "4 4" if null else "none")
        fig.text(x, axis_bottom + 19, f"{tick:g}".replace("-", "−"), 12, g.MUTED, anchor="middle")
    for j, (_, row) in enumerate(table.iterrows()):
        values = forests.parse_effect(row["Without PMV"])
        assert ticks[0] <= values[1] <= values[2] <= ticks[-1]
        estimate, low, high = left + (right-left)*(transform(values)-lo)/(hi-lo)
        y = axis_top + 27 + j*55
        fig.text(600, y+5, row.Contrast, 14, weight=700)
        fig.line(low, y, high, y, g.DARK, 1.8)
        fig.dot(estimate, y, g.DARK)
        fig.text(1000, y+5, row["Without PMV"].replace("-", "−"), 13)


def main():
    data, _ = models.load()
    tables = pd.read_html(forests.SOURCE, encoding="utf-8")
    scales = [
        ([-1.5, -1, -.5, 0, .5, 1, 1.5], False, "Wrist skin temperature difference (°C)"),
        ([.2, .5, 1, 2, 5], True, "Tonic EDA ratio"),
        ([-8, -4, 0, 4, 8], False, "Heart rate difference (bpm)"),
        ([.5, .75, 1, 1.25, 1.5], True, "RMSSD ratio"),
        ([.75, 1, 1.25, 1.5], True, "LF/HF ratio"),
    ]
    fig = g.Figure(1200, 1800)
    fig.text(32, 39, "Physiological responses across four office scenarios", 26, weight=700)
    fig.text(32, 72, "Observed responses and model-estimated levels", 17, weight=700)
    fig.text(600, 72, "Differences relative to T1", 17, weight=700)
    fig.text(32, 98, "Open black circles and bars show model estimates with 95% intervals.", 13, g.MUTED)
    fig.text(600, 98, "Primary model contrasts, averaged equally across morning and afternoon.", 13, g.MUTED)
    for i, ((key, title, unit, *_), ticks, table, scale) in enumerate(zip(
            g.OUTCOMES[:5], combined.TICKS, tables[1:6], scales)):
        top = 137 + 319*i
        if i:
            fig.line(32, top-17, 1168, top-17, "#ccd8df")
        fig.text(32, top, title, 19, weight=700)
        estimates = group_estimates(data, key)
        assert all(ticks[0] <= lo <= hi <= ticks[-1] for _, lo, hi in estimates.values()), key
        combined.physiology(fig, data, key, unit, ticks, top, estimates=estimates)
        pairs = table.loc[table.Contrast.str.endswith("− T1")]
        assert len(pairs) == 3
        contrast_panel(fig, pairs, top, *scale)
    fig.text(32, 1748, "Colored points are individual three-hour session medians. Lines join the same participant within morning or afternoon.", 13, g.MUTED)
    fig.text(32, 1771, "Group estimates average visit order. EDA, RMSSD and LF/HF are back-transformed from their fitted log scales.", 13, g.MUTED)
    g.OUT = forests.OUT
    fig.save("Main physiology - Individual responses and reference contrasts.svg")


if __name__ == "__main__":
    main()
