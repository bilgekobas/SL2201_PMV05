"""Rebuild descriptive exposure and paired-response figures from the submission data."""
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Submission" / "Data in Brief"
OUT = ROOT / "Graphs" / "Reworked Results"
KEY = ["part-id", "scenario", "cluster"]
SCENARIOS = ["T1", "T2", "T3", "T4"]
BLUE, ORANGE, DARK, MUTED = "#176a91", "#ba592c", "#182d38", "#506570"
OUTCOMES = [
    ("skin-temp-wrist-smooth", "Wrist skin temperature", "°C", 4, 1),
    ("eda_tonic", "Tonic EDA", "µS", 16, 4),
    ("hr-clean", "Heart rate", "bpm", 20, 5),
    ("rmssd", "RMSSD", "ms", 100, 25),
    ("lf/hf", "LF/HF ratio", "", 1.2, .3),
    ("tsv", "Thermal sensation vote", "vote points", 4, 1),
    ("tcv", "Thermal comfort vote", "vote points", 4, 1),
]


class Figure:
    def __init__(self, width, height):
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                    f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
                    '<g font-family="Arial, sans-serif">']

    def text(self, x, y, value, size=14, color=DARK, weight=400, anchor="start"):
        self.svg.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
                        f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color, width=1, dash="none"):
        self.svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                        f'stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def dot(self, x, y, color, hollow=False, r=4):
        fill = "white" if hollow else color
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" '
                        f'stroke="{color}" stroke-width="1.2"/>')

    def save(self, filename):
        (OUT / filename).write_text("\n".join([*self.svg, "</g></svg>"]), encoding="utf-8")


def x_axis(fig, left, width, ytop, ybottom, lo, hi, ticks, reference=None):
    def x(value):
        return left + (value - lo) / (hi - lo) * width

    for value in ticks:
        fig.line(x(value), ytop, x(value), ybottom, "#e2e9ec")
        fig.text(x(value), ybottom + 20, f"{value:g}", 12, MUTED, anchor="middle")
    if reference is not None:
        fig.line(x(reference), ytop, x(reference), ybottom, MUTED, 1.3, "4 4")
    fig.line(left, ybottom, left + width, ybottom, "#b6c4cb")
    return x


def draw_exposure(session, window_note="Exposure summaries use the saved in-lab data with breaks excluded."):
    fig = Figure(1120, 505)
    fig.text(36, 43, "Achieved thermal exposure across four office scenarios", 26, weight=700)
    fig.text(36, 70, "Each dot is one participant-session median (11 morning and 11 afternoon per scenario).", color=MUTED)
    fig.dot(42, 96, BLUE)
    fig.text(54, 101, "Morning", 13)
    fig.dot(130, 96, ORANGE, hollow=True)
    fig.text(142, 101, "Afternoon", 13)
    fig.line(235, 96, 263, 96, DARK, 4)
    fig.line(249, 90, 249, 102, DARK, 2)
    fig.text(272, 101, "Median and middle 50% of sessions", 13)
    for left, key, title, lo, hi, ticks, ref in [
        (90, "pmv", "A   Predicted mean vote", .23, .73, np.arange(.3, .71, .1), .5),
        (635, "SET_C", "B   Standard effective temperature (°C)", 25.25, 27.35, np.arange(25.5, 27.1, .5), None),
    ]:
        fig.text(left, 145, title, 17, weight=700)
        x = x_axis(fig, left, 420, 163, 418, lo, hi, ticks, ref)
        if ref is not None:
            fig.text(x(ref), 158, "Design target", 11, MUTED, anchor="middle")
        for i, sc in enumerate(SCENARIOS):
            y = 199 + 62 * i
            fig.text(left - 15, y + 8, sc, 15, weight=700, anchor="end")
            values = session.xs(sc, level="scenario")[key]
            q1, median, q3 = values.quantile([.25, .5, .75])
            fig.line(x(q1), y, x(q3), y, DARK, 4)
            fig.line(x(median), y - 6, x(median), y + 6, DARK, 2)
            for period, color, offset in [("Morning", BLUE, 12), ("Afternoon", ORANGE, 27)]:
                for j, value in enumerate(values.xs(period, level="cluster").sort_values()):
                    fig.dot(x(value), y + offset + (j % 3 - 1) * 3.6, color, period == "Afternoon", 3.3)
    fig.text(36, 466, "T1 Reference     T2 Cooler air / warmer radiant surfaces     T3 Warmer with fan     T4 Warmer with fan / lighter clothing", 13)
    fig.text(36, 487, window_note, 12, MUTED)
    fig.save("Figure 1 - Achieved thermal exposure.svg")


def draw_responses(session, window_note="Physiology uses in-lab session medians. Each dot is one participant. Missing questionnaire pairs are omitted."):
    fig = Figure(1120, 826)
    fig.text(36, 43, "Paired physiological and comfort responses", 26, weight=700)
    fig.text(36, 70, "Scenario minus T1 for the same participant and time of day. Available morning and afternoon differences are averaged.", color=MUTED)
    fig.dot(42, 97, BLUE)
    fig.text(54, 102, "Both times paired", 13)
    fig.dot(177, 97, ORANGE, hollow=True)
    fig.text(189, 102, "One time paired", 13)
    fig.line(320, 91, 320, 103, DARK, 3)
    fig.text(331, 102, "Median participant difference", 13)
    by_scenario = session.unstack("scenario")
    for i, (key, title, unit, extent, step) in enumerate(OUTCOMES):
        left = 83 + i % 2 * 551
        top = 136 + i // 2 * 157
        comparisons = by_scenario[key][SCENARIOS[1:]].sub(by_scenario[key]["T1"], axis=0)
        grouped = comparisons.groupby(level="part-id")
        diffs = grouped.mean()
        counts = grouped.count()
        max_abs = diffs.abs().max().max()
        extent = max(extent, np.ceil(max_abs / step) * step)
        print(key, {sc: (int(diffs[sc].count()), int((counts[sc] == 2).sum()),
                         round(float(diffs[sc].median()), 3)) for sc in diffs})
        panel_title = f"{chr(65 + i)}   {title}" + (f" ({unit})" if unit else "")
        fig.text(left - 46, top, panel_title, 16, weight=700)
        x = x_axis(fig, left, 390, top + 12, top + 110, -extent, extent,
                   np.linspace(-extent, extent, 5), 0)
        for j, sc in enumerate(SCENARIOS[1:]):
            y = top + 30 + j * 28
            fig.text(left - 13, y + 5, sc, 13, anchor="end")
            row = diffs[sc].dropna().sort_values()
            for n, (person, value) in enumerate(row.items()):
                hollow = counts.loc[person, sc] == 1
                fig.dot(x(value), y + (n % 5 - 2) * 3.6, ORANGE if hollow else BLUE, hollow, 3.3)
            fig.line(x(row.median()), y - 11, x(row.median()), y + 11, DARK, 2.6)
            fig.text(left + 401, y + 5, f"n={len(row)}", 12, MUTED)
    fig.text(634, 635, "Reading the vote panels", 16, weight=700)
    fig.text(634, 661, "Positive sensation difference = warmer than T1.", color=MUTED)
    fig.text(634, 683, "Positive comfort-vote difference = less comfortable than T1.", color=MUTED)
    fig.text(36, 800, window_note, 12, MUTED)
    fig.save("Supplement - Paired responses versus T1.svg")


def draw_absolute_responses(
    session, window_note="Physiology shows in-lab session medians (n=11 per cell). Breaks are excluded.",
    vote_note="Votes are observed questionnaire scores. Missing votes leave gaps. Higher comfort scores indicate more discomfort.",
):
    fig = Figure(980, 1470)
    fig.text(36, 43, "Physiological and comfort responses by scenario", 26, weight=700)
    fig.text(36, 70, "Lines connect the same participant across scenarios within each time of day.", color=MUTED)
    fig.text(36, 94, "Black bars show scenario medians. Horizontal offsets separate overlapping observations.", color=MUTED)
    colors = ["#08708b", "#aa6421"]
    periods = ["Morning", "Afternoon"]
    for col, period in enumerate(periods):
        fig.text(298 + col * 443, 133, period, 17, colors[col], 700, "middle")

    participants = session.index.get_level_values("part-id").unique().sort_values()
    offsets = dict(zip(participants, np.linspace(-16, 16, len(participants))))
    for i, (key, title, unit, _, _) in enumerate(OUTCOMES):
        top = 166 + 180 * i
        fig.text(36, top, f"{chr(65 + i)}   {title}" + (f" ({unit})" if unit else ""), 15, weight=700)
        if key in ("tsv", "tcv"):
            ticks = np.arange(-3, 4) if key == "tsv" else np.arange(0, 5)
        else:
            low, high = session[key].min(), session[key].max()
            base = 10 ** np.floor(np.log10((high - low) / 4))
            step = next(base * n for n in [1, 2, 2.5, 5, 10] if base * n >= (high - low) / 4)
            ticks = np.arange(np.floor(low / step), np.ceil(high / step) + 1) * step

        def y(value):
            return top + 124 - (value - ticks[0]) / (ticks[-1] - ticks[0]) * 101

        for col, period in enumerate(periods):
            left = 102 + col * 443
            xs = np.arange(4) * 109 + left + 36
            matrix = session.xs(period, level="cluster")[key].unstack("scenario")[SCENARIOS]
            for tick in ticks:
                fig.line(left, y(tick), left + 380, y(tick), "#e2e9ec")
                fig.text(left - 8, y(tick) + 4, f"{tick:g}", 11, MUTED, anchor="end")
            for person, row in matrix.iterrows():
                points = list(zip(xs + offsets[person], row))
                for (x1, v1), (x2, v2) in zip(points, points[1:]):
                    if pd.notna(v1) and pd.notna(v2):
                        fig.line(x1, y(v1), x2, y(v2), ["#8fb9c5", "#d9b18e"][col])
                for x, value in points:
                    if pd.notna(value):
                        fig.dot(x, y(value), colors[col], r=2.3)
            for x, sc in zip(xs, SCENARIOS):
                median = matrix[sc].median()
                fig.line(x - 12, y(median), x + 12, y(median), DARK, 2.8)
                label = sc if key not in ("tsv", "tcv") else f"{sc}  (n={matrix[sc].count()})"
                fig.text(x, top + 148, label, 11, anchor="middle")
    fig.text(36, 1420, window_note, 12, MUTED)
    fig.text(36, 1441, vote_note, 12, MUTED)
    fig.save("Figure 2 - Absolute physiological and comfort responses.svg")



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    time_series = pd.read_csv(DATA / "SET_timeseries_with_SET.csv")
    exposure = time_series.loc[
        time_series["lab-id"].eq("SL") & time_series["location"].eq("Lab")
        & time_series["break"].ne(1)
    ]
    measures = ["pmv", "SET_C", *[item[0] for item in OUTCOMES[:-2]]]
    session = exposure.groupby(KEY)[measures].median()
    assert len(session) == 88 and session.notna().all().all()
    votes = pd.read_excel(DATA / "00_SL2201_SubjectiveResponses.xlsx", header=1)
    votes = votes[KEY + ["tsv", "tcv"]].dropna(subset=KEY).copy()
    votes["part-id"] = "SL2201_" + votes["part-id"].str.strip()
    votes["cluster"] = votes["cluster"].str.split("_").str[-1]
    votes = votes.set_index(KEY)
    assert len(votes) == 83 and votes.index.is_unique and votes.index.isin(session.index).all()
    draw_exposure(session)
    responses = session.join(votes, validate='one_to_one')
    draw_absolute_responses(responses)
    draw_responses(responses)
    for key in ["pmv", "SET_C"]:
        print(session[key].groupby(level="scenario").agg(["min", "median", "max"]).round(3))


if __name__ == "__main__":
    main()
