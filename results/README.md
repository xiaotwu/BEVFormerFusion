# Experiment records

This directory preserves the numerical content of the former project documentation and adds transcriptions from the accompanying [project report](../reports/From%20Pillars%20to%20Transformers%3A%20Two%20Fusion%20Paradigms%20Based%20on%20PointPillars%20for%203D%20Object%20Detection.pdf).

## Available evidence

| File | Contents | Source |
| --- | --- | --- |
| [metrics.json](metrics.json) | Baseline and Enc_Dec checkpoint curves, seven detection metrics, selected endpoints, and historical workbook hashes | Preserved documentation export |
| [report_results.json](report_results.json) | Tables 5, 6, and 7; hardware, protocol, source PDF hash, and evidence notes | Report Sections 4.2 and 7.2 |
| [per_class_ap.csv](per_class_ap.csv) | Baseline/Enc_Dec class AP at 100k and absolute differences | Report Table 6 |

Original workbooks, raw logs, model checkpoints, and exact baseline/V3 experiment configs are absent. Historical workbook paths and hashes in the export are provenance, not links to available files. Two fusion workbook hashes in that export match and must not be counted as independent trials.

The report's matched lightweight baseline differs from the checked-in R101 reference configuration. Baseline config attribution in the old export has been marked unverified. Report Table 5 rounds scores to three decimals; the preserved baseline/Enc_Dec export retains four decimals. V3 is selected at 90k rather than 100k. Report latency is a transcription, not a newly executed benchmark.

## Checkpoint progression

mAP and NDS are on a 0–1 scale. The table uses shared evaluation milestones; additional baseline milestones and all error terms remain in `metrics.json`.

| Iteration | Baseline mAP | Baseline NDS | Enc_Dec mAP | Enc_Dec NDS |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | 0.1470 | 0.1582 | 0.0898 | 0.1308 |
| 40,000 | 0.1730 | 0.1815 | 0.1271 | 0.1592 |
| 60,000 | 0.1869 | 0.1912 | 0.1785 | 0.1897 |
| 80,000 | 0.1915 | 0.2152 | 0.2066 | 0.2216 |
| 100,000 | 0.2011 | 0.2192 | 0.2507 | 0.2546 |

The selected endpoint maximizes NDS, not necessarily mAP: the baseline's 70k mAP (0.2019) is slightly higher than its 100k mAP (0.2011). Enc_Dec overtakes the baseline at the shared 80k milestone in both headline metrics.

## Interpretation

At 100k, Enc_Dec improves localization and attribute prediction as well as mAP/NDS, while orientation and velocity errors increase. Report class AP improves most for bus, barrier, car, and truck. Its higher accuracy comes with 431.47 ms/sample latency versus 259.32 ms for the camera-only baseline on the stated RTX 3060 machine.

The report gives inconsistent descriptions of repeated-run evidence: Section 4.2.1 mentions repeats within 2%, while Section 7.2 describes single-run results. Without raw runs, seeds, or variance estimates, these summaries do not establish statistical significance or a controlled attribution of gains to any single component.
