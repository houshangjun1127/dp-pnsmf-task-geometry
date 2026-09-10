# Task Geometry Audit for User-Level DP Federated Recommendation

This repository contains the minimum computational materials needed to inspect and rerun the analyses reported in the associated study, including implementation code, frozen configurations, environment specifications, tests, and machine-readable result summaries.

## Version 0.3.0 update

This update adds only the materials required by the revised analysis:

- shared-seed conditional bootstrap summaries;
- complete convergence coverage for the declared dataset, learning-rate, and copy conditions;
- clipping-bound, embedding-dimension, and privacy-budget sensitivity summaries;
- the MovieLens-1M DPALS-style stress test and its stated implementation boundary;
- Netflix-5K5K source, split, availability, and hash records.

## Contents

| Path | Purpose |
|---|---|
| `src/` | P-NSMF, user-level DP, DPALS-style, privacy-accounting, and evaluation implementation |
| `scripts/` | Launchers, audits, and summarizers for the reported stages |
| `tests/` | Unit and protocol tests |
| `configs/` | Frozen protocols and environment locks |
| `results/` | Machine-readable summaries used to check reported values |
| `data/DATASET_MANIFEST.md` | Required inputs, sources, paths, and access limitations |
| `data/UPSTREAM_FILE_HASHES.sha256` | Hashes identifying the audited Amazon Kindle and Netflix-5K5K inputs |
| `THIRD_PARTY_NOTICES.md` | Third-party data and dependency boundaries |
| `CITATION.cff` | Software citation metadata |
| `LICENSE` | MIT License for the authors' original code and documentation |

The repository excludes rendered figures, plotting scripts, raw per-run training histories, user-item interaction files, model factors, manuscripts, and internal research records. These exclusions retain the scope of the initial minimal repository.

## Data boundary

No benchmark interaction data are redistributed. Full retraining requires lawfully obtained files matching `data/DATASET_MANIFEST.md`. The repository records fixed-file locations, counts, and hashes without relicensing the underlying datasets.

## Environment and tests

Use the existing environment locks:

```powershell
conda create --name dp-pnsmf-repro --file configs/conda-explicit-lock.txt
conda activate dp-pnsmf-repro
python -m pytest -q
```

The merged v0.3.0 update passed the retained repository test suite on 10 September 2026 in the existing local scientific environment. This is not an independent clean-machine reproduction.

## Main revision protocols

- Robustness and convergence: `configs/review_revision_sensitivity_uncertainty_v2.0.yaml`
- Dated analysis correction and convergence completion: `configs/review_completion_20260907_v2.1.yaml`
- DPALS-style stress test: `configs/review_revision_dpals_baseline_v2.0.yaml`
- Netflix-5K5K provenance audit: `configs/review_revision_netflix_provenance_v2.0.yaml`

Detailed commands and evidential limits are given in `REPRODUCIBILITY.md`.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The repository URL is https://github.com/houshangjun1127/dp-pnsmf-task-geometry.

The authors' original code and documentation are released under the existing MIT License. Third-party datasets are not included or relicensed, and external dependencies remain under their respective licenses.
