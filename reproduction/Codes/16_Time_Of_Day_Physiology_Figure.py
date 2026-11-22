"""Show fitted time-of-day levels, simple effects and interaction contrasts."""
from html import escape
from importlib import import_module
from itertools import combinations

import numpy as np
from scipy.stats import t

main_fig = import_module("14_Main_Physiology_Figure")
combined, models, g = main_fig.combined, main_fig.models, main_fig.g
PERIODS = combined.PERIODS
COLORS = combined.PERIOD_COLORS
PAIRS = tuple(combinations(g.SCENARIOS, 2))
LABELS = ("Difference (°C)", "Tonic EDA ratio", "Difference (bpm)", "RMSSD ratio", "LF/HF ratio")


def estimate(data, key):
    d = data.dropna(subset=[key])
    design = np.array([models.mixed.row(sc, period == "Afternoon", order, True)
                       for sc, period, order in zip(d.scenario, d.cluster, d.order)])
    beta, cov, _ = models.mixed.gaussian(design, models.response_transform(key, d[key].to_numpy()),
                                        d["part-id"].to_numpy(), d.visit.to_numpy())
    rows = {(sc, period): np.mean([models.mixed.row(sc, period == "Afternoon", order, True)
                                  for order in (1, 2, 3, 4)], axis=0)
            for sc in g.SCENARIOS for period in PERIODS}
    critical = t.ppf(.975, d["part-id"].nunique()-1)

    def effect(w):
        return models.effect((beta, cov), w, np.asarray, critical)

    level = {key_: effect(w) for key_, w in rows.items()}
    within = [effect(rows[sc, "Afternoon"] - rows[sc, "Morning"]) for sc in g.SCENARIOS]
    cross, interaction = [], []
    for a, b in PAIRS:
        morning, afternoon = [rows[b, period] - rows[a, period] for period in PERIODS]
        cross.append([effect(morning), effect(afternoon)])
        interaction.append(effect(afternoon - morning))
    return level, np.array(within), np.array(cross), np.array(interaction)


def tooltip(fig, label, values):
    estimate, low, high = values
    title = escape(f"{label}: {estimate:.3g} [{low:.3g}, {high:.3g}]")
    fig.svg.append(f"<g><title>{title}</title>")


def draw_levels(fig, left, top, level, ticks, key):
    chart_top, chart_bottom = top+58, top+242
    if key in models.paired.LOG:
        back = np.exp
    else:
        back = np.asarray

    def y(value):
        assert ticks[0] <= value <= ticks[-1]
        return chart_bottom - (value-ticks[0])*(chart_bottom-chart_top)/(ticks[-1]-ticks[0])

    for tick in ticks:
        fig.line(left+38, y(tick), left+273, y(tick), "#e0e8ec")
        fig.text(left+30, y(tick)+4, f"{tick:g}", 12, g.MUTED, anchor="end")
    for i, scenario in enumerate(g.SCENARIOS):
        x = left + 60 + 61*i
        for j, (period, color) in enumerate(zip(PERIODS, COLORS)):
            values = back(level[scenario, period])
            est, lo, hi = values
            dot_x = x + (j-.5)*14
            tooltip(fig, f"{scenario} {period}", values)
            fig.line(dot_x, y(lo), dot_x, y(hi), color, 1.8)
            fig.dot(dot_x, y(est), color, hollow=bool(j), r=4)
            fig.svg.append("</g>")
        fig.text(x, chart_bottom+21, scenario, 13, weight=700, anchor="middle")


def draw_contrasts(fig, left, top, values, names, ratio, extent):
    grid_top, grid_bottom = top+56, top+242
    chart_left, width = left + 64, 211

    def x(value):
        return chart_left + width*(value+extent)/(2*extent)

    for tick in np.linspace(-extent, extent, 5):
        null = tick == 0
        fig.line(x(tick), grid_top, x(tick), grid_bottom, g.MUTED if null else "#e0e8ec",
                 1.3 if null else 1, "4 4" if null else "none")
        label = f"{np.exp(tick) if ratio else tick:.2g}".replace("-", "−")
        fig.text(x(tick), grid_bottom+21, label, 12, g.MUTED, anchor="middle")
    for i, (label, group) in enumerate(zip(names, values)):
        center = grid_top+(i+.5)*(grid_bottom-grid_top)/len(names)
        fig.text(left+54, center+4, label, 13, weight=700, anchor="end")
        for j, triple in enumerate(group):
            paired = len(group) == 2
            color = COLORS[j] if paired else g.DARK
            row = center+(j-.5)*10 if paired else center
            est, lo, hi = triple
            assert -extent <= lo <= hi <= extent
            tooltip(fig, f"{label} {PERIODS[j]}" if paired else label, np.exp(triple) if ratio else triple)
            fig.line(x(lo), row, x(hi), row, color, 1.7)
            fig.dot(x(est), row, color, hollow=paired and j == 1, r=3.7)
            fig.svg.append("</g>")


def main():
    data, _ = models.load()
    fig = g.Figure(1320, 1700)
    fig.text(28, 38, "Physiological responses by time of day", 26, weight=700)
    fig.text(28, 66, "Fitted estimates and pointwise 95% intervals from the joint scenario × time-of-day model.", 15, g.MUTED)
    for j, (period, color) in enumerate(zip(PERIODS, COLORS)):
        x = 36+140*j
        fig.dot(x, 93, color, hollow=bool(j), r=4)
        fig.text(x+12, 98, period, 14)
    columns = ((28, "A   Fitted levels", "Original outcome scale"),
               (340, "B   Within each scenario", "Afternoon relative to morning"),
               (656, "C   Within each time of day", "Scenario pairs in morning and afternoon"),
               (990, "D   Change in scenario effects", "Afternoon contrast relative to morning contrast"))
    for left, title, subtitle in columns:
        fig.text(left, 135, title, 17, weight=700)
        fig.text(left, 157, subtitle, 13, g.MUTED)
    names = [f"{b} − {a}" for a, b in PAIRS]
    for i, ((key, title, unit, *_), ticks, scale) in enumerate(zip(g.OUTCOMES, combined.TICKS, LABELS)):
        top = 194+290*i
        if i:
            fig.line(28, top-17, 1287, top-17, "#ccd8df")
        fig.text(28, top, title + (f" ({unit})" if unit else ""), 18, weight=700)
        fig.text(340, top+29, scale, 13, g.MUTED)
        fig.text(656, top+29, scale, 13, g.MUTED)
        ratio = key in models.paired.LOG
        fig.text(990, top+29, "Ratio of scenario ratios" if ratio else "Difference of scenario differences", 13, g.MUTED)
        level, within, cross, change = estimate(data, key)
        maximum = max(np.abs(block).max() for block in (within, cross, change))
        step = .2 if ratio else (.5 if maximum < 4 else 2.)
        extent = np.ceil(maximum/step)*step
        draw_levels(fig, 28, top, level, ticks, key)
        draw_contrasts(fig, 340, top, within[:, None], g.SCENARIOS, ratio, extent)
        draw_contrasts(fig, 656, top, cross, names, ratio, extent)
        draw_contrasts(fig, 990, top, change[:, None], names, ratio, extent)
        for name, values in zip(names, change):
            if values[1] > 0 or values[2] < 0:
                print(title, name, "interaction", *(np.exp(values) if ratio else values))
    fig.text(28, 1661, "The model uses 88 three-hour session medians from 11 participants and adjusts for visit order. Ratio axes are logarithmic.", 13, g.MUTED)
    fig.text(28, 1683, "Panel D directly estimates the interaction using the covariance of the two period-specific contrasts.", 13, g.MUTED)
    g.OUT = main_fig.forests.OUT
    fig.save("Main physiology - Time of day effects.svg")


if __name__ == "__main__":
    main()
