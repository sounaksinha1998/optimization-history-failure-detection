# Correctness-memory experiment track

Parallel stack to residual associative memory. **Classifier training is unchanged.**

## Implementation equations (audit)

**Key (unchanged):**
```
k_t = h_t / (||h_t||_2 + eps)   in R^{d_k}
```

**Target (new):**
```
c_t = 1[y_hat_t = y_t]   in {0, 1}
u_t = c_t P k_t          in R^{d_v}   (P = I when d_v = d_k)
```

**Memory (per level, default d_v = 7):**
```
M^{(j)} in R^{d_v x d_k}
z_t^{(j)} = M^{(j)} k_t
L_M^{(j)} = (1/2) ||M^{(j)} k_t - c_t P k_t||_2^2
M_{t+1}^{(j)} = M_t^{(j)} + eta_j (c_t P k_t - M_t^{(j)} k_t) k_t^T
```

**Deploy readout:**
```
z_j(x) = M_T^{(j)} k_T(x) in R^{d_v}
z_combined(x) = [z_1, ..., z_4] in R^{4 d_v}   (default 28-d)
q(x) = 1 - sigma(w^T z_combined(x) + b)   (cal-fit logistic on error)
```

Configure `AssociativeMemoryConfig(correctness_z_dim=7)`.

## Layout

| Path | Role |
|------|------|
| `optimizer/correctness_memory.py` | Training update + retrieve |
| `research/common/correctness_clinical_training.py` | `train_one_seed_correctness` |
| `research/common/correctness_deployment_pipeline.py` | Deploy `z{j}_d*` features |
| `research/common/correctness_memory_io.py` | `*_correctness_associative.npz` artifacts |
| `probe_experiments/correctness_probe_analysis.py` | Probe protocol on z_combined |
| `notebooks/dermamnist_correctness_memory_full.ipynb` | Combined DermaMNIST notebook |
| `notebooks/_execute_correctness_derma_pipeline.py` | Script runner for notebook |

Residual-memory code under `optimizer/associative_memory.py` and `research/associative_memory_fft/` is **not modified**.

**Note:** Old 512×512 or scalar memory artifacts are incompatible; set `RETRAIN=True` in the notebook after this upgrade.
