# Technical details and reproduction map

This document supplies the persistent-state proof, Private ALS accounting and implementation details, complete-record locators, and the bounded validation-selected comparison. The article retains the core proof, accounting and principal results. The two transferred configurations and their original records remain separate from the additional validation-selected batch.

Main article: Released-Query Diagnostics and Matched Evaluation in Full-Catalog User-Level Differentially Private Federated Recommendation.

Authors: Shangjun Hou and Ruixia Cao.

Repository: [dp-pnsmf-task-geometry](https://github.com/houshangjun1127/dp-pnsmf-task-geometry). Reference numbers follow the main article. References to its propositions, equations and sections are explicitly identified below; S1–S3 denote sections of this supplement.

## S1 Persistent-state privacy proof

### Self contained mechanism definition

Fix the public slot count N, catalog size m, rank d, sampling probability q, clipping radius C>0, noise multiplier sigma>0, number of rounds T, regularization lambda, and data-independent initialization and local/item learning-rate schedules. The schedules may vary publicly with t but cannot be chosen from protected diagnostics without a separately accounted mechanism. Each selected present slot updates its persistent local state from its own record, own participation tape and released shared history, then forms its complete item contribution g. Null slots produce zero and no local update; present-empty records remain present. Define the clipped contribution and releases by

$$
\bar g_{u,t}=g_{u,t}\min\{1,C/\|g_{u,t}\|_F\}.
$$

$$
Y_t=\sum_{u=1}^{N} B_{u,t}\bar g_{u,t}+Z_t,\quad B_{u,t}\sim\mathrm{Bernoulli}(q),\quad Z_t\sim\mathcal{N}(0,\sigma^2C^2I).
$$

$$
V_{t+1}=(1-2\eta_{V,t}\lambda)V_t-\frac{2\eta_{V,t}}{qN}Y_t.
$$

For a zero contribution the clip is defined as zero. Coins and Gaussian draws are fresh, mutually independent and secret. Public normalization is always qN, including adjacent datasets with a null slot. This is the shared update in main article Equation (5); dataset-specific public schedules and parameters are identified by the retained configs and reference-state files. The audited main setting uses q=.1, sigma=3 and T=200. S1 proves main article Proposition 1: the released noisy-aggregate transcript and its shared-model post-processing satisfy the two-direction (ξ, Tρξ)-RDP bound for every integer ξ > 1 under these fixed public parameters; S1.1 gives ρξ and its conversion to (ε, δ)-DP.

The null kernel Q=G0 below is not a premise that the present target's clipped displacement vanishes. Its displacement may be any conditional mixture supported in the C ball. The mixture inequalities and contraction argument handle that general case; zero displacement is only one special member. Hidden-state posteriors across neighboring datasets are not assumed equal.


We use add/remove-user adjacency on a fixed public universe of N slots and m catalog items. Adjacent datasets replace one present record, including its complete interaction history, by a null record in the same slot. Present-empty records remain distinct from null because their missing-item term can contribute. Slot count, catalog and indexing stay fixed across neighbors, as required by the public normalization in main article Equation (5).

The mechanism uses trusted central DP. The curator obtains the unnoised aggregate, generates independent Gaussian noise, and releases the noisy aggregate and subsequent shared item models. The guarantee protects an observer of these releases, not the curator that sees the unnoised sum. It requires the curator to keep raw aggregates, target-user participation, and noise randomness secret and not disclose them to, or collude with, that observer. Local user factors remain privileged to their owners; other clients' views are covered when they depend only on their own records, coins, and the released transcript. Secure aggregation may additionally conceal individual messages, but is neither the source of this DP guarantee nor an implemented cryptographic contribution here.

The protected mathematical release is one ideal training transcript of noisy aggregates and shared models. Participation coins and Gaussian draws remain secret and independent. Offline SNR, raw diagnostics, validation/test metrics, and non-private comparison arms are research measurements with a separate release boundary (main article Section 6.3).

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

Translate by c, divide by C, rotate the shift to one coordinate, and integrate out the identical remaining Gaussian coordinates. A shift of length r=‖u‖/C≤1 is obtained from the unit-shift Gaussian pair by the common channel Kᵣ(x)=rx+√(1−r²)E, where E is an independent N(0,σ²) draw. This channel preserves the noise variance and contracts the mean shift. Rényi data processing in both directions thus reduces the bound to Q₁=N(0,σ²) and P₁=(1−q)Q₁+qN(1,σ²). The sampled-Gaussian reverse moment is no larger than its forward moment [23, Theorem 5 and Corollary 7]. For integer ξ, binomial expansion of the forward Gaussian likelihood-ratio moment gives exp((ξ−1)ρξ) with ρξ as in main article Proposition 1. The preceding two inequalities extend that bound to every hidden-state mixture.

For fixed b, factor the joint output density into conditional kernels. At each common history, the inner conditional moment is at most exp((ξ−1)ρξ); induction gives the following inequality in both directions:

$$
H_\xi(p_{\leq t},q_{\leq t})\leq\exp((\xi-1)\rho_\xi)H_\xi(p_{<t},q_{<t})\leq\exp((\xi-1)t\rho_\xi).
$$

Averaging these conditional moments over the identical auxiliary prior π gives the same bound for (Aⱼ,Y). Dropping Aⱼ is post-processing, proving the actual transcript bound. Although the auxiliary tapes are selected separately for each adjacent pair, the resulting bound is uniform for the actual output. The target tape is never included. Likewise, non-target clients’ own states and messages are functions of their unchanged records, these auxiliary tapes, and Y, so their corresponding views obey the same pairwise bound. This argument requires local state to depend on no other user’s raw data and requires fixed public slot normalization.

Finally, let the total RDP cost be R=Tρξ. The positive part of x−exp(ε), divided by the ξth power of x, is maximized at x=ξexp(ε)/(ξ−1). Applying this maximum to the likelihood ratio and using its moment bound exp((ξ−1)R) gives the following δ bound, whose rearrangement yields main article Proposition 1’s conversion [24]:

$$
\delta\leq(\xi-1)^{\xi-1}\xi^{-\xi}\exp((\xi-1)(R-\epsilon)).
$$

The fixed-slot implementation extends the benchmark update by skipping null local updates and returning zero for null contributions. With all slots present, it uses the identical core update, row order, and random streams without extra draws. Thus the extension preserves the audited all-present utility trajectories, while supplying the neighboring-dataset behavior required for the mathematical statement.

### S1.1 Per-round coefficient and conversion

$$\rho_\xi=\frac{1}{\xi-1}\log\sum_{k=0}^{\xi}\binom{\xi}{k}(1-q)^{\xi-k}q^k\exp\left(\frac{k(k-1)}{2\sigma^2}\right).$$

$$\epsilon=\max\left\{0,\min_{\xi\in\mathcal A}\left[T\rho_\xi+\log(1-1/\xi)-\frac{\log(\delta\xi)}{\xi-1}\right]\right\}.$$

For orders {2,…,64,128,256}, q=0.1, σ=3, T=200 and δ=10⁻⁵, the bound is ε=2.193078238 at order 9. It concerns one secret-randomness transcript. Public-seed simulations, non-private arms, offline metrics and diagnostics have a separate research-release boundary.

## S2 Private ALS implementation and accounting

The practice reference is Google Research’s dp_alternating_minimization at commit f14e9a45dbcc0e8b3cb714a7984c3cefe7660de3 [26]. We follow its implicit-feedback option without centering, floor(βm) frequent-item selection, permanent user tails, local solves from complete frequent-item histories, and capped item contributions. Frequency exponents are fixed to zero (uniform regularization). The second noisy-count query is still executed and charged even though that exponent makes it irrelevant to the regularization weights. The original L1/L5 batch transferred both configurations without target-task selection. The additional bounded six-candidate validation selection is a separate exploratory batch described below. These conventions differ from the 2021 main pseudocode’s mirrored upper-triangle noise and whitening, but follow the declared practice branch. The full-catalog Recall denominator also follows this study’s evaluator rather than the original paper’s capped Recall.

Each count vector has add/remove-user L₂ sensitivity √k and raw coordinate noise standard deviation 100. Given a fixed preceding private transcript, user-factor clipping and the k cap give sensitivities √kΓ², √kΓ, and Γ² for the item-Gram, item-RHS, and shared-Gram channels. Full i.i.d. matrix noise followed by symmetrization has diagonal variance s² and off-diagonal variance s²/2; the corresponding isometric symmetric-vector representation has isotropic variance s². PSD projection and subsequent solves are post-processing. With Γ=1 and the implemented channel scales, the full Gaussian ledger gives (ξ,ξr)-RDP, where

$$
r=\frac{k}{100^2}+\frac{T(2k+1)}{2s^2}=0.12750676012346945.
$$

The first term charges both count queries, and the second charges all three training channels at each iteration. Applying the same improved conversion and integer order set as main article Proposition 1 gives ε=2.193078238 at δ=10⁻⁵ for both L1 and L5. Public outputs are the privately derived item model (with zero rows for rare items) and its post-processing. A user’s final factor is a privileged local output computed from that user’s own training history. The matched arm retains the two noisy count queries and their filtering decisions but removes the three training-noise channels. The complete M output does not have P's reported end-to-end DP guarantee; the paired comparison measures the effect of training noise conditional on shared preprocessing. Table B1 reports descriptive copy-level means; the original CSV files retain all 72 model outcomes and 36 paired losses. The additional batch has 36 private validation candidates and 36 final models forming 18 pairs; its files are separate.

Here T in the Private ALS coefficient denotes the number of ALS iterations (1 or 3), not the 200 rounds of the direct P-NSMF mechanism.

### S2.1 Frozen configurations

| Setting | L1 | L5 |
| --- | ---: | ---: |
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

4. In the private arm independently perturb item Gram, item RHS and shared Gram channels using the frozen noise stream. Symmetrize full matrix draws and project the local and shared Grams separately to PSD before adding ridge. Solve the item systems. The matched arm shares preprocessing, initialization, and the same selected interaction indices but omits the three training-noise channels. The guarantee distinction is stated in S2 above.

5. Restore full catalog rows, using zeros for rare items. Re-solve unclipped evaluation user factors from training histories. Use train-only masking, stable item-ID ties, complete target denominators and the archived full-catalog evaluator. User factors and offline metrics are research outputs, not an additional protected publication.

The source code is [src/training/private_als.py](../src/training/private_als.py); independent event accounting is [src/privacy/private_als_accounting.py](../src/privacy/private_als_accounting.py). The runner and all preprocessing, initialization and training-noise streams are identified in [the reproduction index](R2_SUPPLEMENT_INDEX.md).

### S2.3 Bounded validation selection and complete accounting

| Config | T | Beta | Gravity | Ridge | Training s |
| --- | --- | --- | --- | --- | --- |
| C01 | 1 | 0.05 | 2.0 | 0.5 | 22.314002 |
| C02 | 3 | 0.1 | 0.6 | 0.5 | 38.6489851 |
| C03 | 3 | 0.5 | 0.6 | 0.5 | 38.6489851 |
| C04 | 3 | 1.0 | 0.6 | 0.5 | 38.6489851 |
| C05 | 1 | 0.5 | 0.6 | 0.5 | 22.314002 |
| C06 | 3 | 0.5 | 0.6 | 1.0 | 38.6489851 |

All candidates use rank 32, k = 60, Γ = 1 and raw count noise standard deviation 100. The two count queries give

$$
r_{\mathrm{count}}=\frac{60}{2\cdot100^2}+\frac{60}{2\cdot100^2}=0.006.
$$

Across T sweeps, the item-Gram and item-RHS channels each contribute 60T/(2s²), and the shared-Gram channel contributes T/(2s²). Therefore

$$
r_{\mathrm{train}}=\frac{121T}{2s^2}\simeq0.12150676012346943,\quad r=r_{\mathrm{count}}+r_{\mathrm{train}}\simeq0.12750676012346943.
$$

For an actual Gaussian event with sensitivity Δ, coordinate noise standard deviation a and repetition count n, the contribution is

$$
r_{\mathrm{event}}=\frac{n\Delta^2}{2a^2},\qquad \mathrm{RDP}_{\alpha}=\alpha r.
$$

Each of the 54 private ledgers is checked from these recorded events, not inferred from a saved ε.

The same all-five-query substitution applies to L1/C01 (T = 1, s = 22.314001958810554) and L5/C02 (T = 3, s = 38.64898511285133). C05 uses T = 1 and the C01 scales with β = 0.5 and gravity 0.6. Each private model gives ε = 2.193078238018666 at δ = 10⁻⁵ and order 10, using orders 2 through 64, 128 and 256 and the conversion in S1.1. The main direct P-NSMF mechanism instead has optimal order 9. Float summation can differ in the last digits of r.

M retains the two count queries and the same filtering, selected interaction histories and initialization, and sets the three training sds to zero. The count subtotal is not a privacy guarantee for the complete M output. Removing training noise is not a deletion of privacy cost for that raw-data access. The single-model P bound does not cover the full search, selection or public research archive.

For each task, all six private candidates are evaluated on validation with label 2026102701 and three fixed copies. Absolute validation NDCG20 is averaged equally across copies. Candidates within 10⁻¹² of the maximum form the top tie set; smallest ordered ID wins. Selection is locked before the final 36 models and their unified test. Each selected config uses labels 2026102702,2026102703,2026102704 on each copy with both arms. Validation interactions never enter fitting or user projection. No new candidate test scores are used. Original test data had been visible, so this is exploratory selection in the existing benchmark study, not a confirmation on previously unseen data.

The exact protocol, relative split hashes and streams are in [the portable config](../configs/validation_selected_als.json) and [72-row template](../configs/validation_selected_als_manifest.csv). The complete candidates, lock, 36-row expansion and actual event table are indexed in S3. The ALS comparison retains this study's initial two-task scope of MovieLens and Amazon; Netflix remains included in the three-task P-NSMF audit.

## S3 Copy-level results and reproduction map

Table B1. Private ALS copy-level means over three labels. M/P denote matched/private arms; all quantities are NDCG@20 or Recall@20. Copies are fixed conditions, and no population interval is inferred.

| Dataset / copy / config | M NDCG@20 | P NDCG@20 | Labels | M Recall@20 | P Recall@20 |
| --- | --- | --- | --- | --- | --- |
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

This table keeps the original 12 descriptive copy-level rows. The authoritative model metrics, 36 paired losses, complete ledgers and 12 P-NSMF reference-state identities remain in their existing files; see [R2_SUPPLEMENT_INDEX.md](R2_SUPPLEMENT_INDEX.md). No population-level interval is inferred.

### S3.1 Complete additional batch

The validation table below lists all 12 equal-copy candidate scores. Tables display rounded values for readability; the machine-readable file retains each of the 36 per-copy outcomes without rounding. These are validation scores, not final test results.

Table B2. All candidate validation scores and top tie sets.

| Dataset | Config | Validation NDCG@20 | Top tie | Selected |
| --- | --- | --- | --- | --- |
| amazon_kindle | C01 | 0.000591240176 | False | False |
| amazon_kindle | C02 | 0.000669484346 | False | False |
| amazon_kindle | C03 | 0.000723374360 | False | False |
| amazon_kindle | C04 | 0.000638926076 | False | False |
| amazon_kindle | C05 | 0.000760825052 | True | True |
| amazon_kindle | C06 | 0.000573715134 | False | False |
| ml1m | C01 | 0.0225456886 | True | True |
| ml1m | C02 | 0.0120341351 | False | False |
| ml1m | C03 | 0.00336889041 | False | False |
| ml1m | C04 | 0.00175037914 | False | False |
| ml1m | C05 | 0.00370780643 | False | False |
| ml1m | C06 | 0.00332852290 | False | False |

Final test results use nine pairs per task. Each copy has three independent predeclared labels; copies are equally weighted. Loss is first computed for each pair as

$$
\ell_{\mathrm{pair}}=100\frac{M-P}{M},
$$

and then averaged, rather than replacing it by a ratio of two means. If M is zero the paired percent is undefined and the absolute difference is retained.

Table B3. Final test task means over nine matched pairs.

| Dataset | Config | M NDCG@20 | P NDCG@20 | Mean paired loss (%) | M Recall@20 | P Recall@20 |
| --- | --- | --- | --- | --- | --- | --- |
| amazon_kindle | C05 | 0.033293 | 0.001570 | 95.278516 | 0.034254 | 0.001835 |
| ml1m | C01 | 0.219144 | 0.143305 | 34.544065 | 0.097334 | 0.061106 |

Table B4. Final test copy means over three matched pairs.

| Dataset | Config | Copy | M NDCG@20 | P NDCG@20 | Mean paired loss (%) |
| --- | --- | --- | --- | --- | --- |
| amazon_kindle | C05 | 1 | 0.033378 | 0.001590 | 95.232372 |
| amazon_kindle | C05 | 2 | 0.033903 | 0.001436 | 95.766079 |
| amazon_kindle | C05 | 3 | 0.032597 | 0.001685 | 94.837098 |
| ml1m | C01 | 1 | 0.217428 | 0.144936 | 33.308662 |
| ml1m | C01 | 2 | 0.222608 | 0.144742 | 34.901476 |
| ml1m | C01 | 3 | 0.217395 | 0.140235 | 35.422057 |

The new MovieLens C01 point has substantial ALS loss, not the direct P-NSMF mechanism's small-loss behavior. Amazon C05's private absolute value lies between the old L1/L5 values while its matched value is larger. Old and new seed batches support descriptive comparisons; no cross-algorithm pairing or significance claim is made. Table 8's three tasks, ten shared seeds, 30 pairs per task and Amazon 78.330% retain their own unchanged batch.

Record locations under `results/validation_selected_als/`: `candidate_results.csv` (36), `candidate_selection_summary.csv` (12), `selection_lock.json`, `final_run_manifest.csv` (36), `metrics_by_run.csv` (36), `paired_results.csv` (18), `summary_by_copy.csv` (6), `summary_by_task.csv` (2), `model_identity.csv` (72), `ledger_events.csv` (all 504 actual events), and `ledger_summary.csv` (72). IDs join models, phases and events. `independent_review.json` describes the separate NumPy re-evaluation of 72 saved models, zero maximum metric differences and the report-assembly boundary. No factors or interaction histories are distributed here.

Run `python -m scripts.recompute_validation_selected_als --output NEW_SUMMARY.json` to recompute selection, expansion, paired summaries and actual-event budgets from the distributed records without benchmark files. See [the reproduction guide](../REPRODUCIBILITY.md) for data preparation and the staged scientific runner. Data dictionaries are in [VALIDATION_SELECTED_DATA.md](VALIDATION_SELECTED_DATA.md).

### S3.2 Direct quadratic identity checks

At the reference state (P, V), hold the user-factor matrix P fixed while perturbing the item-factor matrix V. Let x = vec(V), z = vec(ΔV), f(x) = L(P,V)/N, and H be the Hessian of f with respect to x. For user vector pᵤ, item vector vᵢ and observed training items Iᵤ, let a be the missing-item weight and ω the observed-item weight. The local quadratic terms are

$$
A_i=aP^TP+(\omega-a)\sum_{u:i\in I_u}p_up_u^T,\qquad b_i=\omega\sum_{u:i\in I_u}p_u.
$$

Up to terms independent of V, the normalized objective is

$$
f(x)=\mathrm{constant}+\sum_i\left[\frac{v_i^TA_iv_i-2b_i^Tv_i}{N}+\lambda\|v_i\|^2\right].
$$

Its item-gradient and Hessian blocks are

$$
\nabla_i f=\frac{2}{N}(A_iv_i-b_i)+2\lambda v_i,\qquad H_i=\frac{2}{N}A_i+2\lambda I.
$$

At any state, including a nonstationary one, exact quadratic expansion gives

$$
f(x+z)-f(x)=\langle\nabla f(x),z\rangle+\frac{1}{2}z^T H z.
$$

For conditionally zero-mean isotropic noise with covariance s²I, the linear term has zero expectation and

$$
\mathbb{E}[f(x+z)-f(x)\mid P,V]=\frac{s^2}{2}\operatorname{tr}(H).
$$

Here s² is the parameter-update noise variance, not the aggregate's raw σ²C². This calculation does not bound the full alternating/clipping trajectory or top-K utility. The 12 direct full-objective checks (four per task) are in [direct_quadratic_checks.csv](../results/r2_corrections/direct_quadratic_checks.csv) with [summary](../results/r2_corrections/direct_quadratic_summary.json). MovieLens's historical maximum residual 3.82055498349132e-14 and the later direct recomputation 5.800221414276052e-14 are both retained; the difference is floating-point scale. Amazon and Netflix maxima are 8.974429272640139e-15 and 9.842821002692403e-15. The old Table 10 is not rewritten by these later checks. The historical Monte Carlo estimator used 500 independent Hessian-based quadratic forms, equivalent to 500 antithetic pair means with SE denominator √500; these 12 direct checks are not a rerun of that Monte Carlo experiment.

## References

[4] Chien, Steve, Jain, Prateek, Krichene, Walid, Rendle, Steffen, Song, Shuang, Thakurta, Abhradeep, Zhang, Li, “Private Alternating Least Squares: Practical Private Matrix Completion with Tighter Rates,” Proceedings of the 38th International Conference on Machine Learning, vol. 139, pp. 1877-1887, 2021. https://proceedings.mlr.press/v139/chien21a.html

[23] Mironov, Ilya, Talwar, Kunal, Zhang, Li, “Rényi Differential Privacy of the Sampled Gaussian Mechanism,” arXiv:1908.10530, 2019. https://arxiv.org/abs/1908.10530

[24] Canonne, Clément L., Kamath, Gautam, Steinke, Thomas, “The Discrete Gaussian for Differential Privacy,” arXiv:2004.00010, 2020. https://arxiv.org/abs/2004.00010

[26] Google Research, “dp_alternating_minimization,” source revision f14e9a45dbcc0e8b3cb714a7984c3cefe7660de3, accessed 11 October 2026. https://github.com/google-research/google-research/tree/f14e9a45dbcc0e8b3cb714a7984c3cefe7660de3/dp_alternating_minimization
