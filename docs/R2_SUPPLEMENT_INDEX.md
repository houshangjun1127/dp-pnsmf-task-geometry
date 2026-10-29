# Technical supplement and reproduction index

The article retains its core proof, all-five-query accounting and principal results. The complete 0.4.1 baseline and the separate bounded validation-selected records use the same scientific ALS core. See [technical Markdown](R2_TECHNICAL_DETAILS.md) and [PDF](R2_TECHNICAL_DETAILS.pdf).

| Article material | Evidence and executable location |
|---|---|
| Section3.2/4 and S1 | R2_TECHNICAL_DETAILS.md S1; R2_PRIVACY_MODEL.md; src/training/pnsmf_presence.py |
| Section5.1/6.10 and Table 10 | S3.2; results/r2_corrections/direct_quadratic_checks.csv and direct_quadratic_summary.json; retained results/revision_v2/theory_validation/objective_inflation_real_states.json |
| Section6.9 and Table 9 original L1/L5 | S2/S3; configs/private_als_r2_portable_v1.json; results/r2_private_als/metrics.csv,paired_losses.csv,ledger_events.csv |
| Section6.9 and Table 9 validation-selected | configs/validation_selected_als.json; configs/validation_selected_als_manifest.csv; results/validation_selected_als/selection_lock.json and final_run_manifest.csv |
| Every candidate and final outcome | results/validation_selected_als/candidate_results.csv and candidate_selection_summary.csv; metrics_by_run.csv; paired_results.csv; summary_by_copy.csv; summary_by_task.csv |
| Every actual ledger and model identity | results/validation_selected_als/ledger_events.csv,ledger_summary.csv,model_identity.csv |
| Independent validation summary | results/validation_selected_als/independent_review.json and audit_metrics_diff.csv; audit_selection_recomputed.csv; audit_summary_recomputed.csv |
| Data-free arithmetic | python -m scripts.recompute_validation_selected_als --output NEW_SUMMARY.json |
| Staged scientific reproduction | scripts/run_validation_selected_als.py; REPRODUCIBILITY.md; configs/validation_selected_environment.json |
| Field definitions and exclusions | VALIDATION_SELECTED_DATA.md; raw interactions and factors are excluded |

Core reuse at baseline commit32bfcb4633cd47eeda1e8a97cd065f46acfad644:

- `src/training/private_als.py` SHA256 `ddc47192f434a193b3153379078b709c0afe05b7952a0990ca1a0ecbcdc057f8`.
- `src/evaluation/full_catalog_stable.py` SHA256 `098ac5558b7a349084fc3f8531e23b6a27601227301deddd02cb9cab069a4090`.
- `src/privacy/private_als_accounting.py` SHA256 `a8c3a6397fb44501087cb988fe6e2ceee0a81368704adad3ccb672fe1dac44ec`.

The snapshot hashes were checked against the actual execution source, not inferred from filenames. Raw local authorization/recovery logs are not prerequisites for the public commands. The single-model privacy bound is not a guarantee for research metrics, seeds, factors or the entire archive.
