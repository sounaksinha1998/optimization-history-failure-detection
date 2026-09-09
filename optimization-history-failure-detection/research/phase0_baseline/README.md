# Phase 0 — Baseline instrumentation

## Hypothesis

An Adam classifier can be instrumented so that every evaluation example yields
prediction, cross-entropy loss, confidence, and predictive entropy without
changing the optimizer update.

## Experiment

- Dataset: MNIST (`mnist_clean`)
- Optimizer: Adam (lr=0.001)
- Seeds: 42, 123, 456
- Epochs: 5, batch size: 128, train subset: 10000
- Logged fields: epoch, step, sample_id, loss, prediction, correctness,
  confidence, predictive entropy

## Result

- Runs completed: 3
- Mean val accuracy (last epoch): 0.9198
- Mean val loss (last epoch): 0.2729
- Artifacts: `phase0_baseline.csv`, `per_sample.csv`, `metrics.csv`, `plots/`

## Decision

Phase 0 instrumentation is in place. Proceed to Phase 1 (V5B memory as
observation only); do not modify Adam's update.

## Next phase

Phase 1 — maintain multi-timescale memories without changing the Adam step;
write `phase1_memory.csv`.
