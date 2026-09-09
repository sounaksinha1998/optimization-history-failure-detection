# Incremental memory deferral

Tests whether optimization-history retrieval **z(x)** adds routing utility beyond the classifier penultimate representation **h(x)**.

## Methods (same protocol as main MHA deferral)

| Method | Features |
|--------|----------|
| `H` | entropy only |
| `H+h` | entropy + penultimate `h(x)` |
| `H+h+z_actual` | entropy + `h(x)` + MHA retrieval from actual memory |
| `H+h+z_random` | entropy + `h(x)` + MHA retrieval from matched random memory |

## Run

```bash
python -m research.incremental_memory_deferral.incremental_memory_deferral
```

Checkpoints: `final_experiment/checkpoints/` or `memory_identification/artifacts/checkpoints/`.

## Outputs

`research/incremental_memory_deferral/artifacts/`

- `incremental_deferral_per_seed.csv`
- `incremental_deferral_pooled.csv`
- `incremental_deferral_paired_ci.csv`
- `interpretation.md`
