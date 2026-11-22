"""Export the visit-position counts presented in Supplementary Table S3."""

from importlib import import_module
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Codes"))
analysis = import_module("05_AR1_Trajectory_Analysis")


def main():
    raw, _ = analysis.load(clean_eda=True)
    visits = raw[["scenario", "visit", "order"]].drop_duplicates()
    counts = pd.crosstab(visits["scenario"], visits["order"].astype(int))
    assert len(visits) == 44 and counts.shape == (4, 4)
    output = ROOT / "Graphs" / "Reworked Results" / "Reviewer checks"
    output.mkdir(parents=True, exist_ok=True)
    counts.to_csv(output / "visit sequence.csv")


if __name__ == "__main__":
    main()
