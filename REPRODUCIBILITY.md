# Reproduction Guide

## 1. Scope

This guide distinguishes verification from full reproduction. The archived derived results permit direct checking of manuscript values. Full retraining requires the original benchmark files, which are not redistributed here.

## 2. Environment reconstruction

Preferred exact reconstruction:

```powershell
conda create --name dp-pnsmf-repro --file configs/conda-explicit-lock.txt
conda activate dp-pnsmf-repro
```

Fallback package reconstruction:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configs/requirements-lock.txt
```

An exact conda lock may contain platform-specific build URLs. If reconstruction fails on another operating system, create Python 3.11 and install the pinned requirements, then document the deviation.

## 3. Dataset preparation

1. Read `data/DATASET_MANIFEST.md`.
2. Obtain each dataset from an authorized source.
3. Do not substitute another MovieLens release, Amazon category snapshot, or Netflix preprocessing without reporting the change.
4. Verify source filenames, record counts, and available checksums.
5. Apply the archived preprocessing code and fixed-copy protocol.

The experiment launchers expect authorized local copies under `references/vendor/P-NSMF-upstream/data/`. Exact subdirectory and file names are listed in `data/DATASET_MANIFEST.md`. These data paths are excluded from Git tracking.

At freeze, the acquisition and redistribution chain for the fixed Amazon Kindle and Netflix-5K5K files is not sufficiently established for public redistribution. This is a release blocker for the files, not a license to reconstruct or substitute data silently.

## 4. Software verification

From the archive root:

```powershell
python -m pytest -q
```

The pruned public-repository suite contains 88 tests, all of which passed on 17 August 2026 in the existing local scientific environment. Tests verify retained implementation components; they do not by themselves reproduce manuscript results or validate third-party data provenance.

## 5. Main experimental stages

### Frozen main batch

- Configuration: `configs/confirmatory_query_feasibility_v1.0.yaml`
- Launcher: `scripts/run_confirmatory_query_feasibility.py`
- Summarizer: `scripts/summarize_confirmatory_query_feasibility.py`
- Archived output: `results/main_batch/`

### Post-result mechanism analysis

- Configuration: `configs/task_geometry_budget_analysis_v1.0.yaml`
- Launcher: `scripts/run_task_geometry_budget_analysis.py`
- Summarizer: `scripts/summarize_task_geometry_budget_analysis.py`
- Archived output: `results/mechanism_followup/`

### Result-dependent sensitivity analyses

- Configurations: `configs/cross_dataset_lr_cutoff_sensitivity_v1.0.yaml` and `configs/amazon_lr_cutoff_sensitivity_v1.0.yaml`
- Launchers: `scripts/run_cross_dataset_lr_cutoff_sensitivity.py` and `scripts/run_amazon_lr_cutoff_sensitivity.py`
- Summarizers: `scripts/summarize_cross_dataset_lr_cutoff_sensitivity.py` and `scripts/summarize_amazon_lr_cutoff_sensitivity.py`
- Archived output: `results/sensitivity_followup/`

The last two stages were formulated after earlier results had been inspected. Their protocols were fixed before the corresponding follow-up runs, but their findings are robustness, sensitivity, and bounded mechanism evidence rather than independent confirmation.

## 6. Privacy accounting

Use `scripts/audit_privacy_accounting.py` and the accountant interfaces in `src/`. The manuscript retains the more conservative frozen value when independent accountant outputs differ. Any re-run must report the accountant implementation, Rényi-order grid, sampling probability, number of rounds, noise multiplier, and delta.

## 7. Tables and reported values

The minimal repository retains the run-level machine-readable tables and aggregated summaries used to check the manuscript values. Rendered figures and plotting scripts are intentionally excluded because they are not required to rerun the experimental protocols or inspect the reported numerical results.

## 8. Expected resources

The original experiments were CPU runs. Exact wall-clock time depends on data access, operating system, BLAS implementation, and whether the full 200-round matrices or only smoke runs are executed. Before public release, benchmark one representative main-batch run and one geometry run in a clean environment and record time, peak memory, and disk use.

## 9. Acceptable reproduction claims

- Derived numerical audit: supported by the archived summaries.
- Figure regeneration: intended and should be rechecked before release.
- Full computational reproduction: conditional on lawful access to matching benchmark files and successful environment reconstruction.
- Independent replication on alternative datasets: not provided by this archive.

## 10. Repository and citation metadata

 The authors' original code and documentation are released under the MIT License; this license does not relicense third-party datasets or external dependencies. 
