# Memory identification experiments

Isolated **identification / ablation** runs to test whether failure-detection and deferral gains are attributable to **actual optimization-history memory**, versus supervised-gradient statistics, loss, random memory geometry, or penultimate features alone.

- **Code:** `research/memory_identification/memory_identification.py`
- **Outputs:** `research/memory_identification/artifacts/`

Does **not** modify `research/final_experiment/` or the core NRO/NRM mechanism.

## Experiments

### Experiment 1 (label-dependent diagnostic)

| Method | Purpose |
|--------|---------|
| `H`, `H+N_actual` | Existing protocol |
| `H+G` | Gradient-norm control (`G = \|\nabla_\theta \ell\|`) |
| `H+CE` | Supervised-loss attribution control |
| `H+N_rand` | Matched-norm **random frozen memory** (primary identification test) |

### Experiment 2 (label-free MHA deferral)

| Method | Purpose |
|--------|---------|
| `H` | Entropy-only routing |
| `H+h` | Penultimate features (fixed 128-d projection, capacity-matched) |
| `H+z_actual` | Proposed MHA on actual memory |
| `H+z_rand` | MHA on matched random memory |

## Run

```bash
python -m research.memory_identification.memory_identification
```

Options:

- `--no-exp2` — Experiment 1 only
- `--no-exp1` — Experiment 2 only
- `--n-rand 5` — random-memory realizations per seed (default 5)
- `--no-train` — fail if checkpoints missing (default; use trained checkpoints from `final_experiment/checkpoints/` or `artifacts/checkpoints/`)
- `--train-missing` — train missing checkpoints into `artifacts/checkpoints/` only (all Exp1 scores then recomputed from that checkpoint)

## Outputs

1. `experiment1_gradient_memory_controls.csv`
2. `experiment1_per_seed.csv`
3. `experiment1_random_memory_distribution.csv`
4. `experiment1_aggregate.csv`
5. `experiment2_mha_memory_controls.csv`
6. `interpretation.md` — evidence class A / B / C per plan

Checkpoints (if trained) live under `artifacts/checkpoints/` and `artifacts/memory/` only.
