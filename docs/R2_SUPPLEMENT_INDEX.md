# R2 technical supplement and reproduction index

The article retains the compact substantive privacy proof, all-five-query Private ALS accounting, configurations and principal results. This index locates the extended technical materials and unchanged computational evidence.

Equation, section and proposition cross-references in S1–S2 use the associated article's numbering. Numbered literature references in the technical document follow the same article reference list; the four cited sources are reproduced at the end of the technical document.

| Material | Repository file |
|---|---|
| S1 full persistent-state derivation | [Technical details](R2_TECHNICAL_DETAILS.md#s1-persistent-state-privacy-proof), [typeset PDF](R2_TECHNICAL_DETAILS.pdf) |
| S2 algorithm, source differences and full accounting | [Technical details](R2_TECHNICAL_DETAILS.md#s2-private-als-implementation-and-accounting) |
| S3 original 12 copy-level summary rows | [Technical details](R2_TECHNICAL_DETAILS.md#s3-copy-level-results-and-reproduction-map) |
| Source-branch provenance | [R2_ALS_SOURCE.md](R2_ALS_SOURCE.md) |
| Trust and research-release boundaries | [R2_PRIVACY_MODEL.md](R2_PRIVACY_MODEL.md) |
| Fixed protocol and all random streams | [private_als_r2_portable_v1.json](../configs/private_als_r2_portable_v1.json), [run manifest](../configs/private_als_r2_run_manifest_v1.csv) |
| Actual execution and excluded-factor identities | [execution_manifest.csv](../results/r2_private_als/execution_manifest.csv) |
| All 72 model metrics | [metrics.csv](../results/r2_private_als/metrics.csv) |
| All 36 paired comparisons | [paired_losses.csv](../results/r2_private_als/paired_losses.csv) |
| Copy/profile cross-method rows | [comparison_by_copy.csv](../results/r2_private_als/comparison_by_copy.csv) |
| Complete event ledger and budget summary | [ledger_events.csv](../results/r2_private_als/ledger_events.csv), [ledger_summary.csv](../results/r2_private_als/ledger_summary.csv) |
| Twelve reused P-NSMF references and batch identities | [pnsmf_reference_metrics.csv](../results/r2_private_als/pnsmf_reference_metrics.csv), [reference-state configurations](../configs/pnsmf_r2_reference_states.json) |
| Data-free regeneration of summaries and budgets | [recompute_r2_tables.py](../scripts/recompute_r2_tables.py) |
| Arithmetic/fingerprint verification for this documentation | [R2_DOCUMENTATION_CHECK.json](R2_DOCUMENTATION_CHECK.json) |

From the repository root, without interaction data or saved factors:

```text
python -m scripts.recompute_r2_tables --output-dir NEW_R2_CHECK_DIRECTORY
```

The command regenerates paired losses, descriptive summaries, Table 9 and ideal-private budgets into a new directory. Source CSVs stay read-only; matched-arm epsilon is undefined/non-DP. Full retraining instructions and data requirements remain in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md). This documentation update does not perform training, factor evaluation or a clean-machine reproduction.

Code, protocols, metrics, ledgers and random streams are unchanged from the observed repository base. Raw interactions, user/item factors, full training histories, manuscripts, author responses, cover letters and internal coordination records are excluded. Extended proofs and implementation notes are supporting technical documentation, not a substitute for the article's main claims.

Version 0.4.0 / DOI 10.5281/zenodo.23308282 identifies the preceding computational snapshot. The expanded technical files accompany the computational records in this archive and are not part of that preceding snapshot.
