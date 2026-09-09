# Associative Memory + FFT — Implementation (spec correction)

Previous DermaMNIST numbers in this folder were produced under an incorrect trajectory definition (`M_t k_t`, per-sample visit packing, T=1 eval fallback, untrained sample-ID attention). **Do not interpret those results as a test of historical memory.** Retrain after this correction before any scientific claim.

## Exact definitions (current code)

### Associative update (training only)

Independent levels \(M^j \in \mathbb{R}^{d_v \times d_k}\):

\[
k_t = \mathrm{normalize}(h_t(x_t)),\qquad v_t = y_t - p_t
\]

\[
M \leftarrow M - \eta \cdot \frac{1}{B}\sum_{i=1}^{B}(M k_i - v_i)k_i^\top
\]

Per-sample (\(B=1\)) is the same delta rule without averaging. Default \(\eta\) and `update_every` are unchanged. Non-finite \(M\) **fails training**; NaNs are not zeroed at deploy.

Old NRMv2 hierarchical EMA (`optimizer/memory.py`) still runs in observe mode and is **not** used for \(z_{\text{memory}}\).

### Checkpoints

Shared global-step grid `t_1 < … < t_T` (`trajectory_sample_every`, plus the final step). Saved object:

`research/associative_memory_fft/artifacts/memory/{task}_seed{seed}_checkpoints.npz`

with `checkpoint_steps` and `matrices` of shape `(T, K, d_v, d_k)`.

### Trajectory (after training)

Classifier frozen. For every sample in train / cal / test / external:

\[
k_{\text{query}}(x)=\mathrm{normalize}(h_T(x)),\qquad
R[t,j,:](x)=M_t^j k_{\text{query}}(x)
\]

\(R(x)\in\mathbb{R}^{T\times K\times d_v}\) with the **same** \(T,K\) for all groups. Missing checkpoints raise; there is no T=1 fallback.

### FFT

`use_fft=True` (default): `rfft` along the checkpoint time axis of \(R\). Magnitude only. No amplification.

`use_fft=False`: raw \(R\) tokens.

### Attention / aggregation

**Primary experiment: `use_attention=False`.** \(z_{\text{memory}}\) is deterministic mean-pool of \(S\) plus a fixed Gaussian projection (`aggregation_seed`). This is **not** learned attention.

`use_attention=True` requires frozen `attention_params` trained on calibration only. Random / sample-ID \(W_Q,W_K,W_V\) are rejected.

### Random control

\(z_{\text{random}}\): same frozen \(k_{\text{query}}=h_T\), but each \(M_t^j\) is replaced by a Frobenius-norm-matched Gaussian matrix on the same grid, then replayed.

## How to reproduce

1. `pip install -e .`
2. Cheap diagnostics (required before DermaMNIST):  
   `python research/associative_memory_fft/diagnose_associative_memory.py`
3. Full protocol: `notebooks/dermamnist_full_validation.ipynb`  
   Config: `AssociativeMemoryConfig(use_fft=True, use_attention=False)`
4. Cross-dataset probe ($z_{\mathrm{combined}}$ vs $H$): `notebooks/bloodmnist_organcmnist_probe.ipynb`

Compare **H**, **H+h**, **H+h+z_actual**, **H+h+z_random**. Do not treat H vs H+z as the primary question.

## Artifacts (after a run)

| File | Content |
|------|---------|
| `artifacts/spec_correction_diagnostics/` | Cheap synthetic diagnostics |
| `memory/*_checkpoints.npz` | Shared-grid \(M_t\) |
| `memory/*_associative.npz` | Final \(M_T\) |
| `memory_reconstruction_diagnostics.csv` | \(M k\) vs \(v\) on a probe set |

## Scientific readiness

See the correction report in chat. Full DermaMNIST is **not** re-run until unit tests and cheap diagnostics pass. Prior FAST_MODE verdict remains **not scientifically valid** under the intended \(R(x)=M_t k_T\) object.
