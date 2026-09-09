# Memory Implementation Note — Old (Nested EMA) → New (Associative)

This document maps the **frozen V5B nested-EMA memory** touchpoints in the copied baseline to the **associative trajectory memory** replacement. Only memory-related interfaces change; classifier, optimizer (Adam), datasets, and experiment drivers stay frozen.

---

## 1. Old memory semantics (nested EMA)

| Property | Old implementation |
|----------|-------------------|
| **Input signal** | Slow nested signal `A_t` (EMA of batch gradients through G→B→A chain) |
| **State** | `long_term: tuple[pytree, ...]` — one param-tree per level, shape-matched to model params |
| **Update** | Nested EMA: `L^(1)_t = (1-λ₁)L^(1)_{t-1} + λ₁ A_t`; `L^(j)_t = (1-λ_j)L^(j)_{t-1} + λ_j L^(j-1)_t` |
| **Timescales** | Coupled via `compute_long_term_lambdas()` — each level feeds the next |
| **Query (scoring)** | Per-sample gradient `ĝ` dotted with normalized `L̂^(j)` |
| **Output N** | `memory_novelty = 1 - memory_agreement` — **label-dependent** (needs `example_grad`) |
| **Output z** | MHA over flattened `long_term` levels with query `h(x)` |

---

## 2. New memory semantics (associative)

| Property | New implementation |
|----------|---------------------|
| **Key** | `k_t = normalize(h_t(x_t))` at train time; `k_query = normalize(h_T(x))` at deploy |
| **Value** | `v_t = y_t - p_t` (one-hot minus softmax; `d_v = num_classes`) |
| **State** | `AssociativeMemoryState.matrices: tuple[(d_v, d_k), ...]` — K **independent** levels |
| **Update** | Delta rule: `M_{t+1}^j = M_t^j - η_j (M_t^j k_t - v_t) k_t^T` |
| **Timescales** | Independent per level (no nesting): L¹ every 1 step (η=0.1), L² every 4 (η=0.05), L³ every 16 (η=0.01), L⁴ every 64 (η=0.005) |
| **Retrieval** | `L^j(k) = M^j k` — **label-free** at deploy |
| **Checkpoints** | Shared global-step grid `{M_t^j}`; not per-sample visit rows |
| **Trajectory** | After training: `R[t,j,:] = M_t^j normalize(h_T(x))` via `build_query_trajectory` |
| **Spectral** | Optional rFFT along the **checkpoint time** axis → `S(x)` |
| **Output z** | Primary: deterministic mean-pool of S (`use_attention=False`). Learned attention only with frozen cal-trained params |

---

## 3. Touchpoint map: file → symbol → replacement

### 3.1 Core optimizer memory

| Concern | File | Old symbol | New symbol / module |
|---------|------|------------|---------------------|
| Nested EMA update | `optimizer/memory.py` | `long_term_accumulation_step()` | **Unchanged** (V5B observe still runs; not consumed by new scoring) |
| Associative update | `optimizer/associative_memory.py` | — | `AssociativeMemory.update()`, `delta_rule_step()` |
| Associative config | `optimizer/associative_memory.py` | — | `AssociativeMemoryConfig` |
| Random control | `optimizer/associative_memory.py` | — | `randomize_associative_memory()` |
| Optimizer state | `optimizer/v5b.py` | `NRMv2State.long_term` | Side-channel: `AssociativeMemoryState` stored separately (not in NRMv2State) |

**Key design choice:** V5B `long_term` continues to accumulate in observe mode for baseline comparison, but the new scoring path reads only `AssociativeMemoryState` + frozen trajectory.

### 3.2 Training loop

| Concern | File | Old symbol | New symbol / action |
|---------|------|------------|---------------------|
| Training entry | `research/common/clinical_training.py` | `train_one_seed()` | Add associative side-channel inside existing loop (Phase 3) |
| Optimizer factory | `research/common/clinical_training.py` | `make_nrm_v2_optimizer()` | **Unchanged** |
| Save frozen EMA | `research/common/clinical_training.py` | `save_frozen_memory()` | **Kept** for old baseline |
| Save associative | `research/common/memory.py` | — | `save_associative_memory()` / `load_associative_memory()` (Phase 3) |
| Trajectory checkpoints | `research/common/trajectory_store.py` | — | `MemoryCheckpointStore.save_memory_checkpoint()`, `build_query_trajectory(h_T, …)` |
| MSA scoring in eval | `research/common/clinical_training.py` | `evaluate_split(..., score_msa=True)` → `per_sample_msa_signals()` | New path: `per_sample_associative_signals()` (Phase 3) |

**Training side-channel (Phase 3 patch):**

```text
after batch forward (train split only):
  h_batch = resnet18_features(params, x_batch)   # moving h_t
  k_t     = normalize(h_batch)
  v_t     = one_hot(y_batch) - softmax(logits)
  associative_memory.update(k_t, v_t)
  checkpoint_store.save_memory_checkpoint(global_step, M_t)

after training:
  freeze classifier
  k_query = normalize(h_T(x))                    # frozen, all splits
  R(x)    = { M_t^j k_query }                    # shared t-grid
```

### 3.3 Per-sample scoring / signals

| Concern | File | Old symbol | New symbol |
|---------|------|------------|------------|
| Label-dependent N | `research/common/msa.py` | `per_sample_msa_signals(grad, long_term)` | **Kept** for old baseline |
| Label-free z | `research/common/msa.py` | — | `per_sample_associative_signals(h, assoc_state, trajectory, cfg)` |
| Gradient alignments | `research/common/msa.py` | `gradient_memory_alignments()` | Not used in new deploy path |
| h(x) extractor | `research/common/resnet.py` | `resnet18_features()` | **Unchanged** |
| Gradient (old N only) | `research/common/resnet.py` | `example_grad()` | **Unchanged**; not called in new deploy path |

### 3.4 Deferral experiment (z from memory)

| Concern | File | Old symbol | New symbol |
|---------|------|------------|------------|
| Flatten EMA levels | `research/common/msa_linear_deferral.py` | `flatten_memory_matrix(mem_state)` | `spectral_memory.temporal_fft_features(R, use_fft=...)` |
| Load memory matrix | `research/common/msa_linear_deferral.py` | `load_memory_matrix_from_checkpoint()` | Load trajectory + associative state from new artifact path |
| z computation | `research/common/msa_linear_deferral.py` | `compute_z_and_attention(h, M_eff, msa_params)` | `associative_attention.retrieve_z_memory(h, S, use_attention=...)` |
| Linear scorers | `research/common/msa_linear_deferral.py` | `fit_linear_scorers()` | **Unchanged** |
| Deferral metrics | `research/common/msa_linear_deferral.py` | `deferral_metrics()` | **Unchanged** |

### 3.5 Memory identification + random control

| Concern | File | Old symbol | New symbol |
|---------|------|------------|------------|
| Random EMA control | `research/memory_identification/memory_identification.py` | `random_long_term_matched()` | `randomize_associative_memory()` |
| Random matrix control | `research/memory_identification/memory_identification.py` | `random_memory_matrix_matched()` | Norm-matched random `M^j` via `randomize_associative_memory()` |
| Novelty scoring | `research/memory_identification/memory_identification.py` | `compute_novelty_and_grad_norm()` → `per_sample_msa_signals` | Replace with `z_memory` features; keep `N_actual` only for old baseline runs |
| Feature columns | `research/memory_identification/memory_identification.py` | `H`, `h`, `N_actual` | Add `z_memory_*`; `N_actual` optional for comparison |

### 3.6 Diagnostics

| Concern | File | Old symbol | New symbol |
|---------|------|------------|------------|
| V5B state extract | `research/common/memory.py` | `extract_nrm_v2_state()` | **Unchanged** |
| V5B summary | `research/common/memory.py` | `memory_state_summary()` | **Unchanged** |
| Associative extract | `research/common/memory.py` | — | `extract_associative_state()` |
| Reconstruction | `research/common/memory.py` | — | `memory_reconstruction_diagnostics(mem_state, keys, values)` |

### 3.7 Downstream experiment entry points (reuse, patch load sites only)

| Experiment | File | Memory touch |
|------------|------|--------------|
| Training + scoring | `research/common/clinical_failure.py` | Loads scored CSVs; uses `memory_novelty` column today |
| H vs H+N | `research/common/nro_final_experiment.py` | `n_col="memory_novelty"` |
| Deferral | `research/common/msa_linear_deferral.py` | Memory load + z sites |
| H+h+z deferral | `research/incremental_memory_deferral/incremental_memory_deferral.py` | Inherits deferral memory load |
| Memory ID | `research/memory_identification/memory_identification.py` | Memory load + random control |

---

## 4. Label dependence

| Signal | Label-dependent? | Used at deploy? |
|--------|------------------|-----------------|
| `memory_novelty` / `N_actual` | **Yes** — requires `example_grad(params, x, y)` | Old baseline only |
| `memory_agreement`, `c_j`, `alpha_j` | **Yes** — same gradient query | Old baseline only |
| `z_memory` | **No** — `h_T(x)` + frozen trajectory only | New primary signal |
| `H` (normalized entropy) | **No** | Unchanged |
| `h(x)` penultimate features | **No** | Unchanged |

---

## 5. Artifact layout (new runs)

Old EMA artifacts (unchanged path):

```text
research/final_experiment/memory/{task}_seed{seed}_memory.npz
  └── long_term: object array of flattened param-tree vectors per level
```

New associative artifacts (Phase 3+):

```text
research/associative_memory_fft/artifacts/
  memory/{task}_seed{seed}_associative.npz
    └── matrices: (K, d_v, d_k)
  trajectory/{task}_seed{seed}_trajectory.npz
    └── R: (T, K, d_v) — L^j(k_query) over training time
```

---

## 6. Config toggles (new memory only)

```python
@dataclass(frozen=True)
class AssociativeMemoryConfig:
    use_associative_memory: bool = True
    use_fft: bool = True           # False → raw trajectory path
    use_attention: bool = True     # False → mean-pool / linear project
    trajectory_sample_every: int = 1
    num_levels: int = 4
```

---

## 7. Phase status

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 0 — Copy baseline | Done | `docs/COPY_MANIFEST.md` |
| 1 — Inspect EMA | **This document** | Touchpoint map |
| 2 — New modules | In progress | `optimizer/associative_memory.py` (+ trajectory/FFT/attention in subsequent tasks) |
| 3 — Patch touchpoints | Pending | `clinical_training.py`, `memory.py`, `msa.py`, deferral/ID scripts |
| 4–7 — Validate & re-run | Pending | Tests, diagnostics, experiments |
