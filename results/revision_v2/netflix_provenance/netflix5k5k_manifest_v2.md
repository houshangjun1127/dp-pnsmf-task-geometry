# Netflix-5K5K fixed-split provenance audit

The audit passed all frozen invariants. It reconstructed the anonymous positive-event set in memory and wrote no pair data.

| Copy | Train | Validation | Train+validation | Test | Union | Canonical union SHA-256 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 73,320 | 4,616 | 77,936 | 77,936 | 155,872 | `ef4eaca334f8ca2a16f77f3e337114e64ce41cbd625273736a74082af0faadcf` |
| 2 | 73,291 | 4,645 | 77,936 | 77,936 | 155,872 | `ef4eaca334f8ca2a16f77f3e337114e64ce41cbd625273736a74082af0faadcf` |
| 3 | 73,328 | 4,608 | 77,936 | 77,936 | 155,872 | `ef4eaca334f8ca2a16f77f3e337114e64ce41cbd625273736a74082af0faadcf` |

All copy unions identical: **True**.

Primary-source construction: randomly sample 5,000 users and 5,000 items; retain ratings >3 as positive events; split positives equally into training and test; move one training event per eligible user to validation; repeat three times.

Limits: the original raw identifier mapping and sampling seed are not public. Therefore this is an exact audit of the distributed anonymous fixed data object, not a raw-data regeneration claim.
