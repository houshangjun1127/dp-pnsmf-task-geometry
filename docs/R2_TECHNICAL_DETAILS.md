# Technical details for the R2 reproducibility materials

The core privacy argument, algorithm choices, privacy-accounting formula and principal comparison remain in the article. This technical document provides expanded derivations and a reproduction map. It contains no new experiment and changes no result. The matching [PDF](R2_TECHNICAL_DETAILS.pdf) preserves native-equation typesetting. The computational snapshot 0.4.0 is archived at DOI 10.5281/zenodo.23308282; these expanded documents accompany the computational records in this archive and are not part of that preceding snapshot.

## S1 Persistent-state privacy proof

We use add/remove-user adjacency on a fixed public universe of N slots and m catalog items. Adjacent datasets replace one present record, including its complete interaction history, by a null record in the same slot. A null record produces zero contribution and no local update. A present user with no observed positives is distinct from a null record because the missing-item term can still contribute. N, the catalog, and row indexing remain fixed across adjacent datasets; the normalization in Equation (5) is always qN.

The mechanism uses trusted central DP. The curator obtains the unnoised aggregate, generates independent Gaussian noise, and releases the noisy aggregate and subsequent shared item models. The guarantee protects an observer of these releases, not the curator that sees the unnoised sum. It requires the curator to keep raw aggregates, target-user participation, and noise randomness secret and not disclose them to, or collude with, that observer. Local user factors remain privileged to their owners; other clients' views are covered when they depend only on their own records, coins, and the released transcript. Secure aggregation may additionally conceal individual messages, but is neither the source of this DP guarantee nor an implemented cryptographic contribution here.

The protected mathematical release is one ideal training transcript of noisy aggregates and shared models. Participation coins and Gaussian draws remain secret and independent. Offline SNR, raw diagnostics, validation/test metrics, and non-private comparison arms are research measurements with a separate release boundary (Section 6.3).

Fix adjacent datasets differing only at slot j. Let Aⱼ contain the complete Bernoulli participation tapes of all non-target slots. Its prior π is identical on the two datasets. Analyze the auxiliary output (Aⱼ,Y₁,…,Yₜ), where Y denotes the noisy aggregate; this auxiliary output is a proof device, not a proposed release. Condition on Aⱼ=b and a common numerical released history h. Shared item states are fixed by h. Induction over rounds shows that each non-target local factor is the same function of its unchanged record, its fixed participation tape, and h. Hence its current clipped contribution is identical on both sides. Their sum is a common baseline c. A data-independent random initialization can first be conditioned upon and then integrated using its common prior.

On the present-user side, the target’s hidden past state induces a distribution μ of its potential current clipped contribution U, supported on the ball of radius C. The fresh target participation coin is still Bernoulli(q), independent of that past state and the current Gaussian noise. Write Gᵤ for the Gaussian with mean c+u and covariance σ²C²I. The present and null conditional kernels are therefore

$$
P=(1-q)G_0+q\int G_u\,\mu(du),\qquad Q=G_0.
$$

For Pᵤ=(1−q)G₀+qGᵤ, write P=∫Pᵤ μ(du) and repeat the fixed baseline as G₀=∫G₀ μ(du). Define the Rényi moment below, where p and q denote densities. Convexity of the powers with exponents ξ and 1−ξ gives the following two inequalities. This uses the same mixing law to repeat a fixed baseline; it does not equate hidden-state posteriors across datasets.

$$
H_\xi(P,Q)=\int p^\xi q^{1-\xi}=\exp((\xi-1)D_\xi(P\Vert Q)).
$$

$$
H_\xi(P,G_0)\leq\int H_\xi(P_u,G_0)\,\mu(du),
$$

$$
H_\xi(G_0,P)\leq\int H_\xi(G_0,P_u)\,\mu(du).
$$

Translate by c, divide by C, rotate the shift to one coordinate, and integrate out the identical remaining Gaussian coordinates. A shift of length r=‖u‖/C≤1 is obtained from the unit-shift Gaussian pair by the common channel Kᵣ(x)=rx+√(1−r²)E, where E is an independent N(0,σ²) draw. This channel preserves the noise variance and contracts the mean shift. Rényi data processing in both directions thus reduces the bound to Q₁=N(0,σ²) and P₁=(1−q)Q₁+qN(1,σ²). The sampled-Gaussian reverse moment is no larger than its forward moment [23, Theorem 5 and Corollary 7]. For integer ξ, binomial expansion of the forward Gaussian likelihood-ratio moment gives exp((ξ−1)ρξ) with ρξ as in Proposition 1. The preceding two inequalities extend that bound to every hidden-state mixture.

For fixed b, factor the joint output density into conditional kernels. At each common history, the inner conditional moment is at most exp((ξ−1)ρξ); induction gives the following inequality in both directions:

$$
H_\xi(p_{\leq t},q_{\leq t})\leq\exp((\xi-1)\rho_\xi)H_\xi(p_{<t},q_{<t})\leq\exp((\xi-1)t\rho_\xi).
$$

Averaging these conditional moments over the identical auxiliary prior π gives the same bound for (Aⱼ,Y). Dropping Aⱼ is post-processing, proving the actual transcript bound. Although the auxiliary tapes are selected separately for each adjacent pair, the resulting bound is uniform for the actual output. The target tape is never included. Likewise, non-target clients’ own states and messages are functions of their unchanged records, these auxiliary tapes, and Y, so their corresponding views obey the same pairwise bound. This argument requires local state to depend on no other user’s raw data and requires fixed public slot normalization.

Finally, let the total RDP cost be R=Tρξ. The positive part of x−exp(ε), divided by the ξth power of x, is maximized at x=ξexp(ε)/(ξ−1). Applying this maximum to the likelihood ratio and using its moment bound exp((ξ−1)R) gives the following δ bound, whose rearrangement yields Proposition 1’s conversion [24]:

$$
\delta\leq(\xi-1)^{\xi-1}\xi^{-\xi}\exp((\xi-1)(R-\epsilon)).
$$

The fixed-slot implementation extends the benchmark update by skipping null local updates and returning zero for null contributions. With all slots present, it uses the identical core update, row order, and random streams without extra draws. Thus the extension preserves the audited all-present utility trajectories, while supplying the neighboring-dataset behavior required for the mathematical statement.

### S1.1 Per-round coefficient and conversion

$$\rho_\xi=\frac{1}{\xi-1}\log\sum_{k=0}^{\xi}\binom{\xi}{k}(1-q)^{\xi-k}q^k\exp\left(\frac{k(k-1)}{2\sigma^2}\right).$$

$$\epsilon=\max\left\{0,\min_{\xi\in\mathcal A}\left[T\rho_\xi+\log(1-1/\xi)-\frac{\log(\delta\xi)}{\xi-1}\right]\right\}.$$

For orders {2,…,64,128,256}, q=0.1, σ=3, T=200 and δ=10⁻⁵, the bound is ε=2.193078238 at order 9. It concerns one secret-randomness transcript. Public-seed simulations, non-private arms, offline metrics and diagnostics have a separate research-release boundary.

## S2 Private ALS implementation and accounting

The practice reference is Google Research’s dp_alternating_minimization at commit f14e9a45dbcc0e8b3cb714a7984c3cefe7660de3 [26]. We follow its implicit-feedback option without centering, floor(βm) frequent-item selection, permanent user tails, local solves from complete frequent-item histories, and capped item contributions. Frequency exponents are fixed to zero (uniform regularization). The second noisy-count query is still executed and charged even though that exponent makes it irrelevant to the regularization weights. No target-task parameter search is performed. These conventions differ from the 2021 main pseudocode’s mirrored upper-triangle noise and whitening, but follow the declared practice branch. The full-catalog Recall denominator also follows this study’s evaluator rather than the original paper’s capped Recall.

Each count vector has add/remove-user L₂ sensitivity √k and raw coordinate noise standard deviation 100. Given a fixed preceding private transcript, user-factor clipping and the k cap give sensitivities √kΓ², √kΓ, and Γ² for the item-Gram, item-RHS, and shared-Gram channels. Full i.i.d. matrix noise followed by symmetrization has diagonal variance s² and off-diagonal variance s²/2; the corresponding isometric symmetric-vector representation has isotropic variance s². PSD projection and subsequent solves are post-processing. With Γ=1 and the implemented channel scales, the full Gaussian ledger gives (ξ,ξr)-RDP, where

$$
r=\frac{k}{100^2}+\frac{T(2k+1)}{2s^2}=0.12750676012346945.
$$

The first term charges both count queries, and the second charges all three training channels at each iteration. Applying the same improved conversion and integer order set as Proposition 1 gives ε=2.193078238 at δ=10⁻⁵ for both L1 and L5. Public outputs are the privately derived item model (with zero rows for rare items) and its post-processing. A user’s final factor is a privileged local output computed from that user’s own training history. The matched arm shares the private preprocessing but removes training noise, so it is non-private and does not estimate removal of the preprocessing cost. Table B1 reports descriptive copy-level means; the accompanying CSV files retain all 72 model outcomes and 36 paired losses.

Here T in the Private ALS coefficient denotes the number of ALS iterations (1 or 3), not the 200 rounds of the direct P-NSMF mechanism.

### S2.1 Frozen configurations

| Setting | L1 | L5 |
|---|---:|---:|
| Rank | 32 | 32 |
| Interaction cap k | 60 | 60 |
| User clipping radius Γ | 1 | 1 |
| Ridge λ | 0.5 | 0.5 |
| Frequency exponents | 0 | 0 |
| Raw count noise standard deviation | 100 | 100 |
| ALS iterations | 1 | 3 |
| Gravity | 2 | 0.6 |
| Frequent-item fraction β | 0.05 | 0.10 |
| Training base scale s | 22.314001959 | 38.648985113 |
| Target epsilon | 2.193078238 | 2.193078238 |
| Delta | 1e-5 | 1e-5 |

L1/L5 identify the source supplement’s MovieLens-20M ε=1/5 parameter columns, not the budgets of these experiments. Both transferred points were frozen before target-task runs without validation/test tuning or profile selection.

### S2.2 Algorithm

1. For each fixed public user slot, sample at most k observed interactions using its per-slot preprocessing tape. Aggregate binary item counts, add raw Gaussian noise with standard deviation 100, and clamp the released noisy counts below at 1. Rank by noisy count with stable original-item-ID ties; retain floor(βm) frequent items.

2. Keep the complete frequent-item training history for each local solve. Separately select a permanent tail of at most k frequent items by ascending first noisy count, with item-ID ties. Compute the second bounded count vector on those tails, add standard-deviation-100 noise, and clamp below at 1. Charge both counts although frequency exponents are zero.

3. Initialize frequent-item factors once per pair. At every sweep solve each user system from its complete frequent-item history, shared item Gram, gravity and ridge, then clip the solved factor to radius Γ. Form item Gram/RHS statistics using permanent tails; form the shared Gram from all clipped user factors.

4. In the private arm independently perturb item Gram, item RHS and shared Gram channels using the frozen noise stream. Symmetrize full matrix draws and project the local and shared Grams separately to PSD before adding ridge. Solve the item systems. The matched arm shares preprocessing, initialization, and the same selected interaction indices but omits training noise; it remains non-private.

5. Restore full catalog rows, using zeros for rare items. Re-solve unclipped evaluation user factors from training histories. Use train-only masking, stable item-ID ties, complete target denominators and the archived full-catalog evaluator. User factors and offline metrics are research outputs, not an additional protected publication.

The source code is [src/training/private_als.py](../src/training/private_als.py); independent event accounting is [src/privacy/private_als_accounting.py](../src/privacy/private_als_accounting.py). The runner and all preprocessing, initialization and training-noise streams are identified in [the reproduction index](R2_SUPPLEMENT_INDEX.md).

## S3 Copy-level results and reproduction map

Table B1. Private ALS copy-level means over three labels. M/P denote matched/private arms; all quantities are NDCG@20 or Recall@20. Copies are fixed conditions, and no population interval is inferred.

| Dataset / copy / config | M NDCG@20 | P NDCG@20 | Labels | M Recall@20 | P Recall@20 |
|---|---|---|---|---|---|
| MovieLens-1M / 1 / L1 | 0.213261 | 0.141756 | 3 | 0.093661 | 0.060045 |
| MovieLens-1M / 1 / L5 | 0.297998 | 0.075762 | 3 | 0.139514 | 0.030999 |
| MovieLens-1M / 2 / L1 | 0.222861 | 0.139333 | 3 | 0.099299 | 0.060067 |
| MovieLens-1M / 2 / L5 | 0.298128 | 0.084822 | 3 | 0.141510 | 0.035421 |
| MovieLens-1M / 3 / L1 | 0.220462 | 0.154063 | 3 | 0.098287 | 0.065737 |
| MovieLens-1M / 3 / L5 | 0.295041 | 0.091387 | 3 | 0.139121 | 0.037098 |
| Amazon Kindle / 1 / L1 | 0.004932 | 0.001597 | 3 | 0.004436 | 0.001726 |
| Amazon Kindle / 1 / L5 | 0.017315 | 0.001448 | 3 | 0.013960 | 0.001596 |
| Amazon Kindle / 2 / L1 | 0.005794 | 0.001601 | 3 | 0.005019 | 0.001817 |
| Amazon Kindle / 2 / L5 | 0.017123 | 0.001652 | 3 | 0.013727 | 0.001784 |
| Amazon Kindle / 3 / L1 | 0.005236 | 0.001382 | 3 | 0.004693 | 0.001565 |
| Amazon Kindle / 3 / L5 | 0.018023 | 0.001650 | 3 | 0.014092 | 0.001887 |

This table keeps the original 12 descriptive copy-level rows. The authoritative model metrics, 36 paired losses, complete ledgers and 12 P-NSMF reference-state identities remain in their existing files; see [R2_SUPPLEMENT_INDEX.md](R2_SUPPLEMENT_INDEX.md). Neither missing factors nor other seed batches are substituted. No population-level interval is inferred.

## References

[4] Chien, Steve, Jain, Prateek, Krichene, Walid, Rendle, Steffen, Song, Shuang, Thakurta, Abhradeep, Zhang, Li, “Private Alternating Least Squares: Practical Private Matrix Completion with Tighter Rates,” Proceedings of the 38th International Conference on Machine Learning, vol. 139, pp. 1877-1887, 2021. https://proceedings.mlr.press/v139/chien21a.html

[23] Mironov, Ilya, Talwar, Kunal, Zhang, Li, “Rényi Differential Privacy of the Sampled Gaussian Mechanism,” arXiv:1908.10530, 2019. https://arxiv.org/abs/1908.10530

[24] Canonne, Clément L., Kamath, Gautam, Steinke, Thomas, “The Discrete Gaussian for Differential Privacy,” arXiv:2004.00010, 2020. https://arxiv.org/abs/2004.00010

[26] Google Research, “dp_alternating_minimization,” source revision f14e9a45dbcc0e8b3cb714a7984c3cefe7660de3, accessed 11 October 2026. https://github.com/google-research/google-research/tree/f14e9a45dbcc0e8b3cb714a7984c3cefe7660de3/dp_alternating_minimization
