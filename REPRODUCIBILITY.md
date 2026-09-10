# Reproduction guide

## 1. Scope

The archived summaries support direct checking of reported values. Full retraining requires the fixed benchmark files, which are not redistributed. Raw per-run histories are intentionally excluded from this minimal GitHub update.

## 2. Environment

Preferred reconstruction:

```powershell
conda create --name dp-pnsmf-repro --file configs/conda-explicit-lock.txt
conda activate dp-pnsmf-repro
```

Fallback reconstruction:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configs/requirements-lock.txt
```

## 3. Dataset preparation

Obtain authorized copies of the fixed inputs listed in `data/DATASET_MANIFEST.md`. The revision runners accept a data root containing:

```text
PATH_TO_DATA_ROOT/
  ML1M-TXT-FORMAT/
  Amazon_Kindle_Store-TXT-FORMAT/
  Netflix5K5K-TXT-FORAMT/
```

Do not commit the interaction files. Verify the audited Amazon Kindle and Netflix-5K5K inputs against `data/UPSTREAM_FILE_HASHES.sha256` where applicable.

## 4. Tests

From the repository root, run:

```text
python -m pytest -q
```

The merged v0.3.0 update passed the retained test suite on 10 September 2026 in the existing local scientific environment.

## 5. Revision analyses

### Shared-seed robustness, sensitivity, and convergence

```text
python -m scripts.run_review_revision_robustness_protocol --data-root PATH_TO_DATA_ROOT --result-dir NEW_RESULT_DIRECTORY --phase all --workers 4
```

Use a new result directory. The frozen protocols are `configs/review_revision_sensitivity_uncertainty_v2.0.yaml` and `configs/review_completion_20260907_v2.1.yaml`. Archived summary files are under `results/revision_v2/robustness/`.

### DPALS-style stress test

```text
python -m scripts.run_review_revision_dpals_protocol --data-dir PATH_TO_DATA_ROOT/ML1M-TXT-FORMAT --result-dir NEW_DPALS_DIRECTORY --phase all
```

This is an implementation-specific stress test, not a reproduction of a published Private ALS system and not a superiority comparison. The selected validation setting and copy-level test summary are under `results/revision_v2/dpals/`.

### Netflix-5K5K provenance

```text
python -m scripts.audit_netflix5k5k_provenance --data-dir PATH_TO_DATA_ROOT/Netflix5K5K-TXT-FORAMT --output NEW_MANIFEST.json
```

The archived records contain source information, counts, and hashes only. They do not contain user-item interactions.

### Objective-inflation check

```text
python -m scripts.validate_objective_inflation_real_states --help
```

The corresponding summary is `results/revision_v2/theory_validation/objective_inflation_real_states.json`.

## 6. Interpretation limits

- The three fixed copies are not independent samples from a wider recommendation-domain population.
- Shared-seed intervals quantify algorithmic variation conditional on the fixed copies.
- Full reproduction depends on lawful access to matching benchmark files and on the recorded software environment.
- The archive does not establish competitive private recommendation performance or general impossibility outside the declared mechanism and operating points.
