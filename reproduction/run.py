"""Reproduce the revised manuscript analysis from the public Zenodo release."""

import argparse
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
RECORD = "https://zenodo.org/records/16898458/files/"
INPUTS = {
    "00_SL2201_CombinedData.csv": "4ce989c1207f8094e14faf1a117f063fb6b7d6b3f9e672e34a2777d9a96a639a",
    "00_SL2201_SubjectiveResponses.xlsx": "9b0b800d871d17ac9afdeccf253e0e49fab4a42cab52fa2e5fa5deb1d805f7ca",
}
PREFIXES = ("11", "18", "17", "22", "23", "24", "25", "05", "07", "06")


def download_inputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in INPUTS.items():
        path = directory / name
        if not path.exists():
            with urlopen(RECORD + name + "?download=1", timeout=120) as response:
                body = response.read()
            if sha256(body).hexdigest() != digest:
                raise ValueError(f"Published file failed checksum: {name}")
            path.write_bytes(body)
        elif sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input does not match the published file: {path}")


def prepare_timeseries(source, destination):
    import pandas as pd
    from pythermalcomfort.models import set_tmp

    data = pd.read_csv(source)
    data.columns = data.columns.str.lower()
    data = data.loc[data["break"].ne(1)].copy()
    data["tr_C"] = data["mrt"]
    data["rh"] = data["rh"].clip(1, 100)
    columns = ["tdb", "tr_C", "v", "rh", "met", "clo"]
    unique = data[columns].drop_duplicates().copy()
    unique["SET_C"] = set_tmp(
        tdb=unique.tdb.to_numpy(),
        tr=unique.tr_C.to_numpy(),
        v=unique.v.to_numpy(),
        rh=unique.rh.to_numpy(),
        met=unique.met.to_numpy(),
        clo=unique.clo.to_numpy(),
        wme=0,
    ).set
    data = data.merge(unique, on=columns, how="left", validate="many_to_one", sort=False)
    assert len(data) == 15797
    data.to_csv(destination, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    args = parser.parse_args()
    # Generic code generation also works around an LLVM native-target crash on Windows ARM64.
    os.environ.setdefault("NUMBA_CPU_NAME", "generic")
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    download_inputs(args.inputs)
    data = ROOT / "Submission" / "Data in Brief"
    data.mkdir(parents=True, exist_ok=True)
    prepare_timeseries(args.inputs / "00_SL2201_CombinedData.csv", data / "SET_timeseries_with_SET.csv")
    shutil.copyfile(args.inputs / "00_SL2201_SubjectiveResponses.xlsx", data / "00_SL2201_SubjectiveResponses.xlsx")

    outputs = ROOT / "Graphs" / "Reworked Results"
    (outputs / "Revised manuscript").mkdir(parents=True, exist_ok=True)
    codes = ROOT / "Codes"
    for prefix in PREFIXES:
        script, = codes.glob(f"{prefix}_*.py")
        subprocess.run([sys.executable, str(script)], env=environment, check=True)
    for script in ("five_minute_sensitivity.py", "visit_sequence.py"):
        subprocess.run([sys.executable, str(ROOT / script)], env=environment, check=True)


if __name__ == "__main__":
    main()
