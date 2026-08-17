# Task Geometry Audit for User-Level DP Federated P-NSMF

This repository contains the minimum computational materials needed to inspect and rerun the analyses reported in the associated study: implementation code, frozen configurations, environment specifications, tests, and machine-readable result summaries.


## Contents

| Path | Purpose |
|---|---|
| `src/` | P-NSMF, user-level DP training, privacy accounting, evaluation, and task-geometry implementation |
| `scripts/` | Launchers and summarizers for the reported experimental stages |
| `tests/` | Unit and integration tests |
| `configs/` | Frozen protocols and environment locks |
| `results/` | Run-level machine-readable tables and aggregated summaries used in the manuscript |
| `data/DATASET_MANIFEST.md` | Required file names, local placement, and data-access limitations |
| `CITATION.cff` | Software citation metadata |
| `LICENSE` | MIT License covering the authors' original code and documentation |

The repository deliberately excludes rendered figures, plotting scripts, raw per-run JSON histories, user-level diagnostic exports, model factors, manuscripts, research logs, downloaded literature, and third-party datasets. These files are not needed to inspect the reported table values or rerun the declared experimental protocols.

## Data boundary

No benchmark data are redistributed. Full retraining requires lawfully obtained files matching the names and directory structure in `data/DATASET_MANIFEST.md`. The Amazon Kindle and Netflix-5K5K acquisition chain remains insufficiently documented for redistribution, so those files must not be committed.

## Environment and tests

Preferred reconstruction:

```powershell
conda create --name dp-pnsmf-repro --file configs/conda-explicit-lock.txt
conda activate dp-pnsmf-repro
python -m pytest -q
```

Fallback reconstruction:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configs/requirements-lock.txt
python -m pytest -q
```

The pruned minimal repository passed 88 tests on 17 August 2026 in the existing local scientific environment. The suite covers the retained P-NSMF, privacy-accounting, evaluation, and task-geometry components. This is not an independent clean-machine reproduction.

## Experimental stages

- Main paired audit: `configs/confirmatory_query_feasibility_v1.0.yaml`
- Privacy-budget and task-geometry analysis: `configs/task_geometry_budget_analysis_v1.0.yaml`
- Cross-dataset learning-rate and cutoff sensitivity: `configs/cross_dataset_lr_cutoff_sensitivity_v1.0.yaml`
- Amazon-targeted learning-rate sensitivity: `configs/amazon_lr_cutoff_sensitivity_v1.0.yaml`

Detailed commands and evidential limits are provided in `REPRODUCIBILITY.md`.

The latter analyses were designed after the main results had been inspected. They are reported as result-dependent sensitivity and mechanism analyses rather than prospective confirmation.

## Citation, license, and identifiers

Citation metadata are provided in `CITATION.cff`.

- Repository URL: https://github.com/houshangjun1127/dp-pnsmf-task-geometry
- Software license: MIT

The authors' original code and documentation are released under the MIT License. Third-party datasets are not included or relicensed, and external software dependencies remain subject to their respective licenses.
