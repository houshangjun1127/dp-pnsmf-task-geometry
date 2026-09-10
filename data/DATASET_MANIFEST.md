# Dataset manifest

This archive contains no user-item interaction records. The experiments require three fixed copies of MovieLens-1M, Amazon Kindle, and Netflix-5K5K. Users must obtain the files from an authorized source and place them under the directory names below.

## MovieLens-1M

- Source page: https://grouplens.org/datasets/movielens/1m/
- Source archive: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Source archive MD5 used for preparation: `c4d9eecfca2ab87c1945afe126590906`
- Local directory: `ML1M-TXT-FORMAT`
- Expected files: `ML1M-copy{1,2,3}-{train,valid,test}`
- Declared catalog: 6,040 users and 3,952 items.

The main analysis treats observed ratings as implicit events. Fixed files and their use remain subject to the GroupLens terms. This archive does not redistribute them.

## Amazon Kindle

- Fixed-file source: https://github.com/PengQ94/P-NSMF/tree/b900e205d60f219ae3c2d78b2961a5886a50c0a7/data/Amazon_Kindle_Store-TXT-FORMAT
- Audited upstream commit: `b900e205d60f219ae3c2d78b2961a5886a50c0a7`
- Local directory: `Amazon_Kindle_Store-TXT-FORMAT`
- Expected files: `Amazon_Kindle_Store-copy{1,2,3}-{train,valid,test}`
- Declared catalog: 9,862 users and 11,298 items.

The fixed files and their acquisition chain remain subject to the upstream dataset terms. They are not included in this archive.

## Netflix-5K5K

- Construction source: https://www.ijcai.org/Proceedings/13/Papers/396.pdf
- Fixed-file source: https://github.com/PengQ94/P-NSMF/tree/b900e205d60f219ae3c2d78b2961a5886a50c0a7/data/Netflix5K5K-TXT-FORAMT
- Audited upstream commit: `b900e205d60f219ae3c2d78b2961a5886a50c0a7`
- Local directory: `Netflix5K5K-TXT-FORAMT`
- Expected files: `NF5kUsers5kItemsHalfHalf-copy{1,2,3}-{train,valid,test}`
- Declared catalog: 5,000 users and 5,000 items.

The construction source reports sampling 5,000 users and 5,000 items from the Netflix Prize pool, retaining ratings greater than 3, assigning half the positive pairs to training and half to test, moving one eligible training pair per user to validation, and repeating the split three times. The fixed files contain anonymous one-based integer pairs. The raw Netflix identifier mapping and original subset seed are unavailable.

The audit reconstructs 155,872 unique positive pairs in each copy. The three copy unions are identical after canonical sorting and have SHA-256 `ef4eaca334f8ca2a16f77f3e337114e64ce41cbd625273736a74082af0faadcf`. Detailed split counts and file hashes are stored in `results/revision_v2/netflix_provenance/netflix5k5k_manifest_v2.json`.

## File hashes

`UPSTREAM_FILE_HASHES.sha256` lists hashes for the nine Amazon Kindle and nine Netflix-5K5K files at the audited upstream commit. The hash list identifies expected inputs without redistributing their contents.
