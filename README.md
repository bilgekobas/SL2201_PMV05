# Physiological responses to slightly warm office cooling configurations

This repository accompanies a climate-chamber pilot comparing four office cooling configurations designed around a nominal predicted mean vote (PMV) of +0.5. Eleven adults (six male and five female) completed all four visits. Each visit included morning and afternoon sessions on the same date. The revised analysis uses minutes 0–179 of both exposures and keeps all 83 recorded endpoint questionnaires. T1 is the reference for scenario comparisons. Standard effective temperature (SET) is a complementary exposure metric.

## Reproduce the revised analysis

Install Python 3.12 and run these commands from the repository root in an activated virtual environment.

```sh
python -m pip install -r reproduction/requirements.txt
python reproduction/run.py
```

The runner downloads and verifies the checksums of the two files in the [public Zenodo dataset](https://doi.org/10.5281/zenodo.16898458). For an offline run, place the same files in a directory and pass its path with `--inputs PATH`. The runner keeps the archived PMV and mean radiant temperature and regenerates SET with pythermalcomfort 3.4.3. It then runs the session, ordinal, paired continuous-time AR(1) analyses, and categorical 30-minute trajectory models and writes SVG figures, HTML tables and diagnostic files under `reproduction/Graphs/Reworked Results/`. Allow several minutes for the model fits. The categorical output folder includes all six pairwise effects, pointwise 95% intervals, the pooled descriptive Figure 5 and editable SVGs by signal. The original numbered directories are retained for archival reference.

[Submission output map and verification notes](reproduction/VERIFICATION.md)
