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

### Historical DPALS-style stress test (superseded)

```text
python -m scripts.run_review_revision_dpals_protocol --data-dir PATH_TO_DATA_ROOT/ML1M-TXT-FORMAT --result-dir NEW_DPALS_DIRECTORY --phase all
```

Do not use its former same-budget label: its old independent Gram/RHS/shared-Gram accounting was incomplete. Use the R2 comparison below for current baseline evidence. This is an implementation-specific stress test, not a reproduction of a published Private ALS system and not a superiority comparison. The selected validation setting and copy-level test summary are under `results/revision_v2/dpals/`.

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

## 7. R2 validation: no-data checks and frozen simulations

The original planned protocol status is preserved in configs/private_als_r2_portable_v1.json; it is a clearly named path-derived copy, with original SHA256. Actual72-model completion and original excluded-artifact hashes are in results/r2_private_als/execution_manifest.csv. Both profiles and all preprocessing/init/noise entropy vectors remain frozen. Data paths resolve under --data-root; provenance sha256 identities are informational, not local file dependencies.

```text
python -m scripts.recompute_r2_tables --output-dir NEW_R2_CHECK_DIRECTORY
python -m scripts.run_private_als_comparison --help
python -m scripts.run_r2_pnsmf_reference --help
```

The first command generates 36 losses, within-copy summaries, four descriptive groups, 24 cross-method rows, Table 9 and 36 ideal-private budgets from all 72 event ledgers. It verifies the complete archived paired_losses.csv and comparison_by_copy.csv and independently checks event budgets with tolerance 1e-10. Matched epsilon is undefined/non-DP, never 0. Source CSVs are read-only; output directories must be new. summary_descriptive.csv, overview_fixed_copies.csv and table9.json are generated in that output directory; redundant archived copies are omitted. Table 9 is derived from metrics.csv and pnsmf_reference_metrics.csv, with no separate frozen Table 9 comparison claimed by this public command.

For an independently authorized full ALS rerun (not performed during packaging):

```text
python -m scripts.run_private_als_comparison --data-root PATH_TO_DATA_ROOT --output-dir NEW_ALS_RUN_DIRECTORY
```

Use the folder layout in Section 3 and exact18 split hashes in the portable protocol. No interaction files are included. The runner uses one process/at most4 CPU threads, private preprocessing once per pair, frozen rank 32/two profiles and correct raw count sd100. All72 train states are saved before test evaluation. First8 are included in72; only numeric/resource gates allow the other64. Memory monitoring is Windows-specific; --allow-unchecked-memory explicitly waives that gate on other platforms, which have not been validated here. No automatic resume/retry, retuning or test-based selection is provided. Local user factors/diagnostics in generated outputs are research-only. Generated paths and runtime hashes can differ without changing scientific flow.

To rebuild the12 old P-NSMF references, the thin launcher reads every recorded scientific configuration and seed from configs/pnsmf_r2_reference_states.json and invokes the retained run_dp_pnsmf_bgd entry point with new paths. It uses all-present fixed slots, rank 20/T200, the recorded direct gradient/clip and dataset-specific rates/omega; private sigma 3, matched sigma 0. No missing-factor ten-seed batch is substituted:

```text
python -m scripts.run_r2_pnsmf_reference --data-root PATH_TO_DATA_ROOT --output-dir NEW_PNSMF_RUN_DIRECTORY
```

Then evaluate its generated factors with src.evaluation.full_catalog_stable.evaluate_factors using complete original train/test IDs, K20/batch256/CPU float64. The original factor/run SHA and batch identities in pnsmf_reference_metrics.csv identify the old research files, which are not distributed and are not expected to match new compressed-file byte headers. The launcher reconstructs factors; it is not a production DP publisher. Stable evaluator metric conventions replace ambiguous legacy ties, rather than changing training.

Validation scope: the R2 packaging checks were performed against the recorded repository base in the existing local Windows scientific environment. CSV/budget recomputation, rank 32 synthetic original/portable/dense comparisons, null/all-present checks, exact-tie oracle tests and an explicitly enumerated subset of saved factors are recorded separately. No72-model retraining, clean-machine reproduction or other-platform full run is claimed. Existing conda/requirements locks are retained; configs/r2_execution_environment.json records actual versions and limits without installing/upgrading packages.

Corrections and interpretation: docs/R2_CORRECTIONS.md keeps same-batch SNR separate from the frozen three-arm batch, preserves fixed-horizon convergence and the500-independent-quadratic/4-full-objective-per-dataset count. See docs/R2_PRIVACY_MODEL.md for curator trust, persistent-state proof and release exclusions. No manuscript/review letter/internal coordination logs, raw interactions, factors or full training trajectories are included.


## 8. Extended technical documentation

See [R2_SUPPLEMENT_INDEX.md](docs/R2_SUPPLEMENT_INDEX.md) for exact proof, algorithm, ledger and result locations. [R2_TECHNICAL_DETAILS.md](docs/R2_TECHNICAL_DETAILS.md) and its [PDF](docs/R2_TECHNICAL_DETAILS.pdf) contain S1–S3; the main article retains the core argument and total accounting. [R2_DOCUMENTATION_CHECK.json](docs/R2_DOCUMENTATION_CHECK.json) records independent arithmetic checks and hashes of unchanged scientific files at the observed remote commit. This addition changes no training or result. The expanded technical documents accompany the computational records in this archive and are not part of the preceding version 0.4.0 snapshot.
