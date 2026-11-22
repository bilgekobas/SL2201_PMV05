# Submission output map and verification

## Manuscript

| Item | Generated file or source |
| --- | --- |
| Tables 1–2 | Experimental design targets and instrument interpretation, recorded in the manuscript |
| Table 3 | Scenario summary in `Revised manuscript/Manuscript figures and tables.html` |
| Table 4 | T1 contrasts and T4 versus T2 in `Revised manuscript/Achieved PMV model comparison.html` |
| Figure 1 | Original chamber schematic and photograph, a static manuscript asset |
| Figures 2–4 | Visit schedule, pooled exposure and pooled physiology SVGs in `Reviewer checks/` |
| Figure 5 | `Revised manuscript/Final physiology - Pooled AR1 trajectories.svg` |
| Figure 6 | `Revised manuscript/Main physiology - Annotated time-of-day comparisons.svg` |

## Supplement

| Item | Generated file or source |
| --- | --- |
| Table S1 | Instrument records and manufacturer specifications summarized in the supplement |
| Tables S2–S3 | Questionnaire coverage in `Revised manuscript/Manuscript figures and tables.html` and visit counts in `Reviewer checks/visit sequence.csv` |
| Figure S1 and Tables S4–S5, S7 | `Reviewer checks/Session model diagnostics.svg` and `reviewer diagnostic values.json` |
| Table S6 | `Reviewer checks/five-minute slope sensitivity.csv` |
| Tables S8–S10 | `Revised manuscript/Achieved PMV model comparison.html` |
| Figures S2–S3 and Tables S11–S12 | `Revised manuscript/Effect-size figures and tables/` |
| Figure S4 and Table S15 | Slope interaction SVG and tables in `Revised manuscript/AR1 physiology - Time-of-day trajectory analysis.html` |
| Figure S5 | `AR1 climate and physiology trajectories.svg` |
| Tables S13–S14 | `Reviewer checks/session sensitivity checks.json` |
| Figure S6 | `Revised manuscript/Final physiology - Morning and afternoon AR1 trajectories.svg` |
| Table S16 | `Revised manuscript/Physiology time-of-day interaction table.html` |

Paths are relative to `reproduction/Graphs/Reworked Results/`. Word contains raster exports of the plots. The pipeline generates the SVG sources. The experiment schematic and photograph, design targets and equipment descriptions are documentary items.

## Verification

The archived submission copy was rerun independently of the original results directory. Results were compared with the numbers displayed in Word and the corresponding SVG sources. The public workbook produced the same reconciled endpoint records. SET was regenerated from the public time series and matched the archived SET for every retained row on the tested environment. Exact floating-point agreement can vary with platform and optimizer iteration. In the public-input rerun the PMV-adjusted TSV T4 versus T2 upper endpoint was 34.590 compared with 34.591 in the archived run. The displayed central estimate (6.330) and lower endpoint (1.159) matched. Some diagnostic optimization quantities also varied at small numerical scales.
