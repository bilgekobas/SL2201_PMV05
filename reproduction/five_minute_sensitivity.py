"""Export the five-minute trajectory checks presented in Supplementary Table S6."""

from importlib import import_module
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Codes"))
analysis = import_module("18_AR1_Time_Of_Day_Interaction")


def main():
    _, frames = analysis.ar.load(clean_eda=True)
    thinned = frames.loc[frames["time-elapsed"].mod(5).eq(0)]
    rows = []
    for key, label in analysis.ar.paired.FAMILIES["Physiology"].items():
        _, full_slopes, _, _ = analysis.fit(frames, key)
        _, thin_slopes, _, diagnostics = analysis.fit(thinned, key)
        for full_row, thin_row in zip(full_slopes, thin_slopes, strict=True):
            pair = full_row[0]
            assert pair == thin_row[0]
            if pair == "T4 − T2" or (key == "lf/hf" and pair in {"T3 − T2", "T4 − T3"}):
                rows.append([label, pair, *full_row[-1], *thin_row[-1], diagnostics["Rows"]])
    columns = ["Outcome", "Pair", "Full estimate", "Full low", "Full high",
               "Five-minute estimate", "Five-minute low", "Five-minute high", "Timestamps"]
    assert len(rows) == 7
    output = ROOT / "Graphs" / "Reworked Results" / "Reviewer checks"
    pd.DataFrame(rows, columns=columns).to_csv(output / "five-minute slope sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
