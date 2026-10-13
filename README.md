# Task Geometry Audit for User-Level DP Federated Recommendation

This repository contains the minimum computational materials needed to inspect and rerun the analyses reported in the associated study, including implementation code, frozen configurations, environment specifications, tests, and machine-readable results.

## Extended technical documentation

The associated R2 article is titled “Released-Query Diagnostics and Matched Evaluation in Full-Catalog User-Level Differentially Private Federated Recommendation.” The repository retains its historical name.

The [technical supplement and reproduction index](docs/R2_SUPPLEMENT_INDEX.md) adds the expanded persistent-state proof, declared Private ALS algorithm and accounting, and original copy-level summaries. Read the [Markdown](docs/R2_TECHNICAL_DETAILS.md) or [typeset PDF](docs/R2_TECHNICAL_DETAILS.pdf). The article retains its core privacy argument, accounting formula and principal comparison.

This is a documentation-only addition. Training code, frozen configurations, all 72 models/36 pairs, random streams and results are unchanged. [Documentation checks](docs/R2_DOCUMENTATION_CHECK.json) independently recompute the budgets and bind existing scientific files to the observed remote base. No retraining or new factor evaluation is claimed. The expanded documents accompany the computational records in this archive and are not part of the preceding version 0.4.0 snapshot.

## Version 0.4.1 — R2 update

Version 0.4.0 adds the complete two-profile Private ALS comparison:

- Results for all 72 models and 36 paired comparisons.
- Portable frozen protocols, execution records, and preprocessing, initialization, and training-noise random streams.
- Complete privacy-accounting event ledgers.
- Stable full-catalog evaluation and necessary correctness tests.
- Evaluation results, identities, and reconstruction configurations for 12 existing P-NSMF reference states.
- Tools to regenerate descriptive summaries and Table 9 and independently check privacy budgets.
- Clarifications of the privacy model, statistical interpretation, and theory-validation scope.

Both literature-based parameter migrations are calibrated to epsilon = 2.193078238018666 and delta = 1e-5. The labels L1 and L5 identify the source configurations; they are not the privacy budgets used in this comparison.

The new Private ALS comparison supersedes the historical DPALS-style stress test as current baseline evidence. The old code and results remain available for traceability, but their former same-budget accounting was incomplete and must not be reused as a fair-comparison premise.

See the [corrections](docs/R2_CORRECTIONS.md), [implementation sources and differences](docs/R2_ALS_SOURCE.md), and [trusted-central privacy model](docs/R2_PRIVACY_MODEL.md).

### Check R2 summaries and budgets without benchmark data

From the repository root:

```text
python -m scripts.recompute_r2_tables --output-dir NEW_R2_CHECK_DIRECTORY
python -m pytest -q
```

The recomputation command reads the archived model metrics, P-NSMF reference metrics, frozen protocol, and event ledgers. It checks the complete paired and cross-method result tables and independently recomputes the private-model budgets.

Per-copy descriptive summaries, the four-group overview, and Table 9 are generated in the new output directory rather than stored as redundant result files. Complete model metrics and all paired and cross-method rows remain archived.

Full rerunning commands are provided in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Benchmark interaction data and model factors are excluded.

### Interpretation and release boundaries

The comparison describes the declared operating points conditional on the fixed data copies. It does not establish competitive optimality, independent causal effects, or a general impossibility result for private recommendation. Both configurations and all results remain included.

Each ALS matched arm shares the realized private preprocessing and initialization with its private arm and removes only training noise. Its output is not DP; its privacy budget must not be described as epsilon = 0.

The direct P-NSMF mechanism is analyzed under a trusted-central privacy model with fixed public user slots, null adjacency, and a persistent-state argument. The privacy guarantee concerns an ideal single execution with secret randomness and the specified protected release. Public seeds, user factors, private diagnostics, nonprivate controls, and jointly released multiple-run archives are outside that guarantee.

These materials are a public-seed research simulation archive, not a protected production release.

The current associated revision title is “Released-Query Diagnostics and Matched Evaluation in Full-Catalog User-Level Differentially Private Federated Recommendation”. Historical releases and tags remain unchanged. Version 0.4.0 / DOI 10.5281/zenodo.23308282 identifies the preceding computational snapshot and does not include the expanded technical documents.

## Historical version 0.3.0 update

Version 0.3.0 added:

- Shared-seed conditional bootstrap summaries.
- Complete convergence coverage for the declared dataset, learning-rate, and copy conditions.
- Clipping-bound, embedding-dimension, and privacy-budget sensitivity summaries.
- The MovieLens-1M DPALS-style stress test and its stated implementation boundary.
- Netflix-5K5K source, split, availability, and hash records.

The historical stress test remains available, with its evidential limitations clarified by the R2 update.

## Contents

| Path | Purpose |
|---|---|
| `src/` | P-NSMF, Private ALS, privacy accounting, stable evaluation, and retained historical implementations |
| `scripts/` | Running, auditing, and summary/budget recomputation |
| `tests/` | Necessary implementation and protocol checks |
| `configs/` | Frozen protocols, random streams, reference-state configurations, and environment records |
| `results/r2_private_als/` | Complete model metrics, paired results, reference metrics, execution identities, and accounting ledgers |
| `results/r2_corrections/` | Compact statistical and theory-scope correction evidence |
| `results/revision_v2/` | Historical revision summaries |
| `docs/` | R2 privacy model, implementation sources, corrections, and interpretation limits |
| `data/DATASET_MANIFEST.md` | Required inputs, sources, paths, and access limitations |
| `data/UPSTREAM_FILE_HASHES.sha256` | Hashes identifying the audited Amazon Kindle and Netflix-5K5K inputs |
| `CHANGELOG.md` | Version changes |
| `THIRD_PARTY_NOTICES.md` | Third-party data and dependency boundaries |
| `CITATION.cff` | Software citation metadata |
| `LICENSE` | MIT License for the authors' original code and documentation |

The repository excludes rendered figures, plotting scripts, raw per-run training histories, user-item interaction files, model factors, manuscripts, author responses, review letters, and internal research records.

## Data boundary

No benchmark interaction data are redistributed. Full retraining requires lawfully obtained files matching [data/DATASET_MANIFEST.md](data/DATASET_MANIFEST.md).

The R2 portable protocol records the exact fixed-split hashes. Data paths resolve under the explicitly supplied data root. The repository records input identities and provenance without relicensing the underlying datasets.

## Environment and validation

Use the existing environment locks:

```powershell
conda create --name dp-pnsmf-repro --file configs/conda-explicit-lock.txt
conda activate dp-pnsmf-repro
python -m pytest -q
```

The historical v0.3.0 update passed the retained repository test suite on 10 September 2026 in the existing local scientific environment.

R2 packaging checks included merged repository tests, CSV summary and budget recomputation, rank-32 synthetic consistency checks, null/all-present checks, exact-tie evaluation tests, and evaluation checks on eight saved ALS states. After removing redundant aggregate files, the revised CSV-only entry was checked again against the accepted summaries and Table 9.

These checks did not include formal model retraining, reevaluation of all 72 saved models during packaging, clean-machine reproduction, or a full run on another operating system.

The recorded R2 environment is provided in `configs/r2_execution_environment.json`. Existing dependency locks are retained. Windows-specific memory monitoring in the ALS runner is documented in the reproduction guide.

## Main revision protocols

- R2 Private ALS: `configs/private_als_r2_portable_v1.json`
- R2 frozen 72-model run list and random streams: `configs/private_als_r2_run_manifest_v1.csv`
- P-NSMF reference-state reconstruction: `configs/pnsmf_r2_reference_states.json`
- Historical robustness and convergence: `configs/review_revision_sensitivity_uncertainty_v2.0.yaml`
- Dated analysis correction and convergence completion: `configs/review_completion_20260907_v2.1.yaml`
- Historical, superseded DPALS-style stress test: `configs/review_revision_dpals_baseline_v2.0.yaml`
- Netflix-5K5K provenance audit: `configs/review_revision_netflix_provenance_v2.0.yaml`

Detailed commands and evidential limits are given in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Citation and license

Software citation metadata are provided in [CITATION.cff](CITATION.cff).

Repository: https://github.com/houshangjun1127/dp-pnsmf-task-geometry

The authors' original code and documentation are released under the existing MIT License. Third-party datasets are not included or relicensed, and external dependencies remain under their respective licenses.
