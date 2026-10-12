# Trusted-central fixed-slot privacy model and persistent-state argument

This note covers the direct_item_gradient / clip mechanism with mathematical real arithmetic and ideal secret independent randomness. It is not a floating-point/PRNG deployment certificate. The curator is trusted and can observe raw records, persistent user states, contributions and unnoised aggregates. The guarantee protects recipients of the shared transcript, not that curator. No cryptographic secure aggregation implementation is claimed.

Fix public N user slots, catalog M, rank, initialization and parameters. A slot is null (zero contribution and no local update) or present with its full finite binary history, including a legitimate empty history. Neighbors exchange one present record and null; this is add/remove, not replacement between two nonnull histories. N/catalog/indices never shrink. Empty-present is not null: missing-item terms can make its direct gradient nonzero. Local updates depend only on that user's record, previous state, current sampling coin and common released item history. After updating a selected present user, clip its complete item gradient to Frobenius norm C.

Fresh coins B[u, t] are mutually independent Bernoulli(q); fresh Z[t] are independent N(0, sigma^2 C^2 I), independent of data, initialization and coins. The curator releases Y[t]=sum of selected present clipped gradients+Z[t]. The item update uses the public factor 2*eta[t]/(qN), public regularization and Y[t]. The protected view is public inputs, Y[1: T] and their shared-model postprocessing. User factors, raw aggregates, private diagnostics/SNR/Hessian/objectives, raw holdout metrics, participation disclosures, noise/seeds, nonprivate arms and jointly released multiple runs are excluded. Independently public initialization may be conditioned on without altering this statement.

## Why persistent user states do not invalidate the conditional bound

For a neighboring pair whose target slot is j, augment the hypothetical output with every non-target user's entire independent coin tape A[j]. Both sides have the same prior law for A[j]. Do not assume its posterior given a shared history is equal under both datasets; it generally is not. Do not disclose the target tape or the Gaussian tape, which would destroy the sampling argument.

Fix one non-target tape b and the same numerical released history h. Induction on local updates makes all non-target states and their current aggregate baseline c(b, h) identical. The target present-side hidden past state has an arbitrary conditional law. Its potential selected contribution U lies in the C-ball. Its fresh current coin remains Bernoulli(q), independent of that past and the fresh noise. Therefore the two current kernels are

P=(1-q)G0+q integral Gu mu(du), Q=G0,

where Gu=N(c+u, sigma^2 C^2 I). The null baseline can be repeated under the very same present-side mixing measure mu; this does not require a null-side target posterior.

For order a>1, H_a(P, Q)=integral p^a q^(1-a). Write P_u=(1-q)G0+qGu. Jensen with convex x^a gives H_a(P, G0)<=integral H_a(P_u, G0)mu(du). Separately, convex x^(1-a) gives H_a(G0, P)<=integral H_a(G0, P_u)mu(du). These are moment inequalities in both directions, not a claim that Renyi divergence itself is jointly convex.

After translation/scaling/rotation, any shift norm r<=1 reduces to the one-dimensional Gaussian pair with shift r. Apply the common data-independent channel K_r(x)=r*x+sqrt(1-r^2)*E, E~N(0, sigma^2), to the shift1 pair. It produces exactly the shift r pair with unchanged noise variance. Renyi data processing bounds both directions by the boundary pair. Thus the hidden-state mixture has a history-uniform bound rho_a=max(D_a(M1||phi0), D_a(phi0||M1)), with M1=(1-q)phi0+qphi1 and phi_s=N(s, sigma^2).

For fixed b, factor the joint density into conditional kernels and integrate the current kernel last. Each H_a moment is at most exp((a-1)rho_a); induction gives exp((a-1)T*rho_a). Both directions obey the same induction. The augmented joint density has the same tape prior pi(b) on both sides, so its moment averages these conditional bounds under pi. Discard A[j] by data processing. This argument can use a different augmentation for each target pair without publishing all users' tapes together. Freely evolving neighboring trajectories need not stay C-close; only common-history conditional kernels are compared.

For integer a>=2 the Gaussian boundary pair has the valid bidirectional SGM bound

rho_a=log(sum[k=0..a] binom(a, k)(1-q)^(a-k)q^k exp(k(k-1)/(2*sigma^2)))/(a-1).

The reverse boundary moment is bounded by the forward one as in [Mironov, Talwar and Zhang (2019), Section 3.1](https://arxiv.org/abs/1908.10530). The hidden-mixture reverse direction was handled separately above. Conditional composition/data processing also follow [Mironov (2017)](https://arxiv.org/abs/1702.07476).

## Actual conversion and all-present extension

Given total RDP R_a, ordinary conversion is R_a+log(1/delta)/(a-1). The tighter conversion used here is

epsilon_a=R_a+log(1-1/a)-log(delta*a)/(a-1).

Indeed maximize (x-exp(epsilon))_+/x^a at x=a*exp(epsilon)/(a-1); integrate under Q and use E_Q L^a<=exp((a-1)R_a). This gives the displayed conversion in every ordered direction. See also [Canonne, Kamath and Steinke (2020), Proposition12](https://arxiv.org/abs/2004.00010). Minimize over public integer orders2..64, 128, 256. With q=.1, sigma 3, T200, delta1e-5 it gives epsilon 2.193078238018666 (optimal order9); ordinary conversion is approximately2.569082681744053.

The presence adapter keeps all N rows, filters only absent selected slots, draws no randomness, and calls the existing direct-gradient core. With all presence entries true the selected order, dimensions, arithmetic and RNG state are exactly unchanged. Induction preserves the all-present trajectory. Small synthetic checks verify this invariant and empty/null behavior; they do not certify all historical files or production privacy. Data-dependent parameters, public participation, cross-user raw-state dependencies, secret-noise disclosure or compaction would require a new analysis.

For Private ALS each fixed-preprocessing clipped statistic uses its recorded Gaussian sensitivity/noise and conditional composition. Its two counts and all three channels per sweep are in ledger_events.csv. The resulting single-model sufficient bound does not imply the whole research archive is DP.
