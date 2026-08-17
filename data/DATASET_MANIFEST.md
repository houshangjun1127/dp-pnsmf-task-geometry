# Dataset manifest

This repository does not contain raw or preprocessed benchmark data. The frozen launchers expect lawfully obtained P-NSMF-format files under the following local paths.

## MovieLens-1M

- Local directory: `references/vendor/P-NSMF-upstream/data/ML1M-TXT-FORMAT/`
- Declared users/items: 6,040/3,952
- Required files for each `copy` in `{1,2,3}`:
  - `ML1M-copy{copy}-train`
  - `ML1M-copy{copy}-valid`
  - `ML1M-copy{copy}-test`
- Dataset identity: MovieLens-1M. The official archive is available from GroupLens at `https://grouplens.org/datasets/movielens/1m/`.
- Important limitation: the experiments use the fixed P-NSMF-format copies rather than a newly reconstructed split from the official ratings archive.

## Amazon Kindle

- Local directory: `references/vendor/P-NSMF-upstream/data/Amazon_Kindle_Store-TXT-FORMAT/`
- Declared users/items: 9,862/11,298
- Required files for each `copy` in `{1,2,3}`:
  - `Amazon_Kindle_Store-copy{copy}-train`
  - `Amazon_Kindle_Store-copy{copy}-valid`
  - `Amazon_Kindle_Store-copy{copy}-test`
- Provenance status: the manuscript follows fixed files associated with the P-NSMF reference implementation. The original rating-to-event threshold and redistribution chain are not independently verifiable from the current research record.
- Release rule: do not redistribute these files.

## Netflix-5K5K

- Local directory: `references/vendor/P-NSMF-upstream/data/Netflix5K5K-TXT-FORAMT/`
- Declared users/items: 5,000/5,000
- Required files for each `copy` in `{1,2,3}`:
  - `NF5kUsers5kItemsHalfHalf-copy{copy}-train`
  - `NF5kUsers5kItemsHalfHalf-copy{copy}-valid`
  - `NF5kUsers5kItemsHalfHalf-copy{copy}-test`
- `TXT-FORAMT` retains the historical upstream spelling used by the frozen launchers.
- Provenance status: Netflix-5K5K is a historical preprocessed subset associated with the P-NSMF reference implementation, not an external validation dataset. Its acquisition and redistribution chain is not independently verified in the current research record.
- Release rule: do not redistribute these files.

## Fixed-copy interpretation

Each dataset has three fixed train/validation/test copies representing split-and-seed conditions within the same source. They are not independent samples from a recommendation-domain population. Any substitute files or reconstructed splits constitute a protocol deviation and must be reported.
